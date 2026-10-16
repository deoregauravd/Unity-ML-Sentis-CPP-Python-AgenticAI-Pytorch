using System;
using System.Collections.Generic;
using Unity.Mathematics;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
#if UNITY_ANDROID
using UnityEngine.Android;
#endif

/// <summary>
/// Complete Standalone Canvas UI Hand & Gesture Tracking System.
/// Runs 100% on standard Unity UI (ScreenSpaceOverlay) and WebCamTexture.
/// Works on all Android phones and PC without requiring ARCore or XR plugins.
/// </summary>
public class CanvasHandGestureController : MonoBehaviour
{
    [Header("Sentis AI Models (Auto-loaded from Resources if empty)")]
    public ModelAsset handDetectorAsset;
    public ModelAsset handLandmarkerAsset;
    public TextAsset handAnchorsCSV;

    [Header("Detection Thresholds")]
    [Range(0.1f, 1f)] public float handScoreThreshold = 0.45f;
    [Range(0.1f, 1f)] public float handIouThreshold = 0.3f;
    public int maxHands = 2;

    [Header("Camera Configuration")]
    public int requestedWidth = 1280;
    public int requestedHeight = 720;
    public int requestedFPS = 30;
    public bool preferFrontCamera = false;
    public int manualRotationAngle = 0;

    [Header("Color Palette")]
    public Color hand0Color = new Color(0f, 0.95f, 1f, 0.95f);    // Cyan
    public Color hand1Color = new Color(1f, 0.15f, 0.65f, 0.95f);  // Magenta
    public Color boneColor = new Color(1f, 1f, 1f, 0.6f);

    // MediaPipe Hand Constants
    const int k_NumAnchors = 2016;
    const int k_DetectorInputSize = 192;
    const int k_LandmarkerInputSize = 224;
    const int k_NumLandmarks = 21;
    const int k_MaxHands = 2;

    static readonly (int, int)[] HandBones = new (int, int)[]
    {
        (0, 1), (1, 2), (2, 3), (3, 4),        // Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        // Index
        (5, 9), (9, 10), (10, 11), (11, 12),   // Middle
        (9, 13), (13, 14), (14, 15), (15, 16), // Ring
        (13, 17), (17, 18), (18, 19), (19, 20),// Pinky
        (0, 17)                                // Palm Base
    };

    // UI Elements (Built automatically at runtime)
    Canvas m_Canvas;
    CanvasScaler m_CanvasScaler;
    RawImage m_CameraFeedRawImage;
    RectTransform m_CameraFeedRect;
    AspectRatioFitter m_CameraFitter;

    // Start / Permission Dialog
    GameObject m_PermissionDialog;
    Text m_DialogTitle;
    Text m_DialogMessage;
    Button m_StartButton;

    // HUD Elements
    GameObject m_HUDPanel;
    Text m_StatusText;
    Text m_GestureSummaryText;
    Button m_SwitchCameraButton;
    Button m_RotateButton;

    // Hand Visuals on Canvas
    RectTransform[] m_HandRoots = new RectTransform[k_MaxHands];
    RectTransform[][] m_JointMarkers = new RectTransform[k_MaxHands][];
    RectTransform[][] m_BoneLines = new RectTransform[k_MaxHands][];
    Text[] m_GestureBadges = new Text[k_MaxHands];
    RectTransform[] m_GestureBadgeRects = new RectTransform[k_MaxHands];

    // Inference State
    WebCamTexture m_WebCamTexture;
    int m_ActiveCameraDeviceIndex = 0;
    float[,] m_Anchors;
    Worker m_DetectorWorker;
    Worker m_LandmarkerWorker;
    Tensor<float> m_DetectorInputTensor;
    Tensor<float> m_LandmarkerInputTensor;

    bool m_IsRunning;
    bool m_ModelsLoaded;
    int m_FpsCount;
    float m_FpsTimer;
    int m_CurrentFps;

    void Awake()
    {
        BuildCanvasUI();
    }

    void Start()
    {
        ShowPermissionDialog();
    }

    void Update()
    {
        // Monitor FPS
        m_FpsCount++;
        m_FpsTimer += Time.deltaTime;
        if (m_FpsTimer >= 1f)
        {
            m_CurrentFps = m_FpsCount;
            m_FpsCount = 0;
            m_FpsTimer = 0f;
        }

        // Keep Camera Aspect Ratio and Orientation updated
        if (m_WebCamTexture != null && m_WebCamTexture.isPlaying)
        {
            UpdateCameraFeedTransform();
        }

        // Keyboard hotkeys for editor testing
        if (Keyboard.current != null)
        {
            if (Keyboard.current.rKey.wasPressedThisFrame)
            {
                CycleCameraRotation();
            }
            if (Keyboard.current.cKey.wasPressedThisFrame)
            {
                SwitchCamera();
            }
        }
    }

    void OnDestroy()
    {
        m_IsRunning = false;
        if (m_WebCamTexture != null)
        {
            m_WebCamTexture.Stop();
            Destroy(m_WebCamTexture);
        }
        m_DetectorWorker?.Dispose();
        m_LandmarkerWorker?.Dispose();
        m_DetectorInputTensor?.Dispose();
        m_LandmarkerInputTensor?.Dispose();
    }

    #region Canvas UI Generation

    void BuildCanvasUI()
    {
        // 1. Root Canvas
        var canvasObj = new GameObject("HandGestureCanvas");
        m_Canvas = canvasObj.AddComponent<Canvas>();
        m_Canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        m_Canvas.sortingOrder = 100;

        m_CanvasScaler = canvasObj.AddComponent<CanvasScaler>();
        m_CanvasScaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        bool isPortrait = Screen.height > Screen.width;
        m_CanvasScaler.referenceResolution = isPortrait ? new Vector2(1080, 1920) : new Vector2(1920, 1080);
        m_CanvasScaler.matchWidthOrHeight = isPortrait ? 0f : 1f;

        canvasObj.AddComponent<GraphicRaycaster>();

        // Ensure EventSystem exists
        if (FindAnyObjectByType<UnityEngine.EventSystems.EventSystem>() == null)
        {
            var es = new GameObject("EventSystem", typeof(UnityEngine.EventSystems.EventSystem), typeof(UnityEngine.InputSystem.UI.InputSystemUIInputModule));
        }

        // 2. Camera Background RawImage (Stretches to entire screen)
        var bgObj = new GameObject("CameraFeed", typeof(RectTransform), typeof(CanvasRenderer), typeof(RawImage), typeof(AspectRatioFitter));
        bgObj.transform.SetParent(m_Canvas.transform, false);
        m_CameraFeedRect = bgObj.GetComponent<RectTransform>();
        m_CameraFeedRect.anchorMin = Vector2.zero;
        m_CameraFeedRect.anchorMax = Vector2.one;
        m_CameraFeedRect.pivot = new Vector2(0.5f, 0.5f);
        m_CameraFeedRect.sizeDelta = Vector2.zero;
        m_CameraFeedRect.anchoredPosition = Vector2.zero;

        m_CameraFeedRawImage = bgObj.GetComponent<RawImage>();
        m_CameraFeedRawImage.color = Color.black; // Starts black until camera is on

        m_CameraFitter = bgObj.GetComponent<AspectRatioFitter>();
        m_CameraFitter.aspectMode = AspectRatioFitter.AspectMode.EnvelopeParent;

        // 3. Hand Visualizers Layer (Stretches over screen)
        for (int h = 0; h < k_MaxHands; h++)
        {
            var handObj = new GameObject($"HandLayer_{h}", typeof(RectTransform));
            handObj.transform.SetParent(m_Canvas.transform, false);
            m_HandRoots[h] = handObj.GetComponent<RectTransform>();
            m_HandRoots[h].anchorMin = Vector2.zero;
            m_HandRoots[h].anchorMax = Vector2.one;
            m_HandRoots[h].pivot = new Vector2(0.5f, 0.5f);
            m_HandRoots[h].sizeDelta = Vector2.zero;
            m_HandRoots[h].anchoredPosition = Vector2.zero;

            Color themeColor = (h == 0) ? hand0Color : hand1Color;

            // Bones (21 UI line segments)
            m_BoneLines[h] = new RectTransform[HandBones.Length];
            for (int b = 0; b < HandBones.Length; b++)
            {
                var boneObj = new GameObject($"Bone_{b}", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
                boneObj.transform.SetParent(m_HandRoots[h], false);
                var rt = boneObj.GetComponent<RectTransform>();
                rt.anchorMin = Vector2.zero;
                rt.anchorMax = Vector2.zero;
                rt.pivot = new Vector2(0.5f, 0.5f);
                var img = boneObj.GetComponent<Image>();
                img.color = boneColor;
                m_BoneLines[h][b] = rt;
                boneObj.SetActive(false);
            }

            // Joints (21 UI circle dots)
            m_JointMarkers[h] = new RectTransform[k_NumLandmarks];
            for (int j = 0; j < k_NumLandmarks; j++)
            {
                var jointObj = new GameObject($"Joint_{j}", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
                jointObj.transform.SetParent(m_HandRoots[h], false);
                var rt = jointObj.GetComponent<RectTransform>();
                rt.anchorMin = Vector2.zero;
                rt.anchorMax = Vector2.zero;
                rt.pivot = new Vector2(0.5f, 0.5f);
                float dotSize = (j == 0 || j == 4 || j == 8 || j == 12 || j == 16 || j == 20) ? 22f : 16f;
                rt.sizeDelta = new Vector2(dotSize, dotSize);
                var img = jointObj.GetComponent<Image>();
                img.color = (j == 4 || j == 8 || j == 12 || j == 16 || j == 20) ? Color.white : themeColor;
                m_JointMarkers[h][j] = rt;
                jointObj.SetActive(false);
            }

            // Gesture Badge over Hand
            var badgeObj = new GameObject($"GestureBadge_{h}", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(Outline));
            badgeObj.transform.SetParent(m_HandRoots[h], false);
            m_GestureBadgeRects[h] = badgeObj.GetComponent<RectTransform>();
            m_GestureBadgeRects[h].anchorMin = Vector2.zero;
            m_GestureBadgeRects[h].anchorMax = Vector2.zero;
            m_GestureBadgeRects[h].pivot = new Vector2(0.5f, 0.5f);
            m_GestureBadgeRects[h].sizeDelta = new Vector2(240f, 54f);

            var bgImg = badgeObj.GetComponent<Image>();
            bgImg.color = new Color(0.08f, 0.1f, 0.15f, 0.88f);
            var outline = badgeObj.GetComponent<Outline>();
            outline.effectColor = themeColor;
            outline.effectDistance = new Vector2(2, -2);

            var textObj = new GameObject("BadgeText", typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
            textObj.transform.SetParent(badgeObj.transform, false);
            var textRt = textObj.GetComponent<RectTransform>();
            textRt.anchorMin = Vector2.zero;
            textRt.anchorMax = Vector2.one;
            textRt.sizeDelta = Vector2.zero;
            m_GestureBadges[h] = textObj.GetComponent<Text>();
            m_GestureBadges[h].font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
            m_GestureBadges[h].fontSize = 22;
            m_GestureBadges[h].alignment = TextAnchor.MiddleCenter;
            m_GestureBadges[h].color = themeColor;
            m_GestureBadges[h].text = "Hand Detecting...";

            badgeObj.SetActive(false);
        }

        // 4. Top HUD Header
        m_HUDPanel = new GameObject("HUDPanel", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        m_HUDPanel.transform.SetParent(m_Canvas.transform, false);
        var hudRt = m_HUDPanel.GetComponent<RectTransform>();
        hudRt.anchorMin = new Vector2(0f, 1f);
        hudRt.anchorMax = new Vector2(1f, 1f);
        hudRt.pivot = new Vector2(0.5f, 1f);
        hudRt.anchoredPosition = Vector2.zero;
        hudRt.sizeDelta = new Vector2(0f, 140f);
        m_HUDPanel.GetComponent<Image>().color = new Color(0.05f, 0.08f, 0.12f, 0.75f);

        // Status Text in HUD
        var statusObj = new GameObject("StatusText", typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        statusObj.transform.SetParent(m_HUDPanel.transform, false);
        var sRt = statusObj.GetComponent<RectTransform>();
        sRt.anchorMin = new Vector2(0f, 0.5f);
        sRt.anchorMax = new Vector2(0.7f, 1f);
        sRt.anchoredPosition = new Vector2(25f, -10f);
        sRt.sizeDelta = new Vector2(-50f, 0f);
        m_StatusText = statusObj.GetComponent<Text>();
        m_StatusText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
        m_StatusText.fontSize = 20;
        m_StatusText.alignment = TextAnchor.MiddleLeft;
        m_StatusText.color = Color.white;
        m_StatusText.text = "Initializing MediaPipe Hand Tracking...";

        // Gesture Summary Text
        var gestSumObj = new GameObject("GestureSummary", typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        gestSumObj.transform.SetParent(m_HUDPanel.transform, false);
        var gRt = gestSumObj.GetComponent<RectTransform>();
        gRt.anchorMin = new Vector2(0f, 0f);
        gRt.anchorMax = new Vector2(0.7f, 0.5f);
        gRt.anchoredPosition = new Vector2(25f, 5f);
        gRt.sizeDelta = new Vector2(-50f, 0f);
        m_GestureSummaryText = gestSumObj.GetComponent<Text>();
        m_GestureSummaryText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
        m_GestureSummaryText.fontSize = 24;
        m_GestureSummaryText.alignment = TextAnchor.MiddleLeft;
        m_GestureSummaryText.color = new Color(1f, 0.85f, 0.2f, 1f);
        m_GestureSummaryText.text = "Gesture: Show your hand";

        // Flip Camera Button
        var camBtnObj = CreateButton("SwitchCamBtn", m_HUDPanel.transform, "Flip Cam", new Vector2(1f, 0.5f), new Vector2(-170f, 0f), new Vector2(130f, 54f));
        m_SwitchCameraButton = camBtnObj.GetComponent<Button>();
        m_SwitchCameraButton.onClick.AddListener(SwitchCamera);

        // Rotate Sensor Button
        var rotBtnObj = CreateButton("RotateBtn", m_HUDPanel.transform, "Rotate", new Vector2(1f, 0.5f), new Vector2(-25f, 0f), new Vector2(120f, 54f));
        m_RotateButton = rotBtnObj.GetComponent<Button>();
        m_RotateButton.onClick.AddListener(CycleCameraRotation);

        // 5. Permission & Start Dialog Card
        BuildPermissionDialog();
    }

    void BuildPermissionDialog()
    {
        m_PermissionDialog = new GameObject("PermissionDialog", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        m_PermissionDialog.transform.SetParent(m_Canvas.transform, false);
        var pRt = m_PermissionDialog.GetComponent<RectTransform>();
        pRt.anchorMin = new Vector2(0.5f, 0.5f);
        pRt.anchorMax = new Vector2(0.5f, 0.5f);
        pRt.pivot = new Vector2(0.5f, 0.5f);
        pRt.sizeDelta = new Vector2(680f, 440f);
        pRt.anchoredPosition = Vector2.zero;

        var img = m_PermissionDialog.GetComponent<Image>();
        img.color = new Color(0.08f, 0.12f, 0.18f, 0.95f);

        var outline = m_PermissionDialog.AddComponent<Outline>();
        outline.effectColor = new Color(0f, 0.9f, 1f, 0.8f);
        outline.effectDistance = new Vector2(3, -3);

        // Title
        var titleObj = new GameObject("Title", typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        titleObj.transform.SetParent(m_PermissionDialog.transform, false);
        var tRt = titleObj.GetComponent<RectTransform>();
        tRt.anchorMin = new Vector2(0f, 0.7f);
        tRt.anchorMax = new Vector2(1f, 1f);
        tRt.anchoredPosition = new Vector2(0f, -20f);
        m_DialogTitle = titleObj.GetComponent<Text>();
        m_DialogTitle.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
        m_DialogTitle.fontSize = 32;
        m_DialogTitle.alignment = TextAnchor.MiddleCenter;
        m_DialogTitle.fontStyle = FontStyle.Bold;
        m_DialogTitle.color = new Color(0f, 0.95f, 1f, 1f);
        m_DialogTitle.text = "MediaPipe Hand Gestures";

        // Message
        var msgObj = new GameObject("Message", typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        msgObj.transform.SetParent(m_PermissionDialog.transform, false);
        var mRt = msgObj.GetComponent<RectTransform>();
        mRt.anchorMin = new Vector2(0.08f, 0.35f);
        mRt.anchorMax = new Vector2(0.92f, 0.72f);
        mRt.anchoredPosition = Vector2.zero;
        m_DialogMessage = msgObj.GetComponent<Text>();
        m_DialogMessage.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
        m_DialogMessage.fontSize = 22;
        m_DialogMessage.alignment = TextAnchor.MiddleCenter;
        m_DialogMessage.color = new Color(0.9f, 0.9f, 0.95f, 1f);
        m_DialogMessage.text = "This application tracks your hands and recognizes gestures in real-time using on-device AI.\n\nTap below to enable your camera.";

        // Start Button
        var btnObj = CreateButton("StartButton", m_PermissionDialog.transform, "Start Camera & Tracking", new Vector2(0.5f, 0.2f), Vector2.zero, new Vector2(380f, 75f));
        m_StartButton = btnObj.GetComponent<Button>();
        m_StartButton.onClick.AddListener(OnStartButtonClicked);

        var btnText = btnObj.GetComponentInChildren<Text>();
        if (btnText != null)
        {
            btnText.fontSize = 24;
            btnText.fontStyle = FontStyle.Bold;
        }
    }

    GameObject CreateButton(string name, Transform parent, string label, Vector2 anchor, Vector2 pos, Vector2 size)
    {
        var btnObj = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(Button), typeof(Outline));
        btnObj.transform.SetParent(parent, false);
        var rt = btnObj.GetComponent<RectTransform>();
        rt.anchorMin = anchor;
        rt.anchorMax = anchor;
        rt.pivot = anchor;
        rt.anchoredPosition = pos;
        rt.sizeDelta = size;

        var img = btnObj.GetComponent<Image>();
        img.color = new Color(0.12f, 0.45f, 0.85f, 0.95f);

        var outline = btnObj.GetComponent<Outline>();
        outline.effectColor = new Color(0.3f, 0.8f, 1f, 0.9f);
        outline.effectDistance = new Vector2(2, -2);

        var textObj = new GameObject("Text", typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        textObj.transform.SetParent(btnObj.transform, false);
        var tRt = textObj.GetComponent<RectTransform>();
        tRt.anchorMin = Vector2.zero;
        tRt.anchorMax = Vector2.one;
        tRt.sizeDelta = Vector2.zero;

        var txt = textObj.GetComponent<Text>();
        txt.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
        txt.fontSize = 18;
        txt.alignment = TextAnchor.MiddleCenter;
        txt.color = Color.white;
        txt.text = label;

        return btnObj;
    }

    #endregion

    #region Permission & Camera Control

    void ShowPermissionDialog()
    {
        if (m_PermissionDialog != null)
            m_PermissionDialog.SetActive(true);
    }

    async void OnStartButtonClicked()
    {
        m_StartButton.interactable = false;
        m_DialogMessage.text = "Requesting camera permission...";

#if UNITY_ANDROID
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Permission.RequestUserPermission(Permission.Camera);
            int waitFrames = 60;
            while (!Permission.HasUserAuthorizedPermission(Permission.Camera) && waitFrames > 0)
            {
                await Awaitable.WaitForSecondsAsync(0.1f);
                waitFrames--;
            }
        }

        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            m_DialogMessage.text = "Camera permission was denied!\nPlease enable camera in App Settings.";
            m_StartButton.interactable = true;
            return;
        }
#endif

        m_DialogMessage.text = "Initializing Neural Networks & Camera Feed...";
        await Awaitable.WaitForSecondsAsync(0.05f);

        InitializeModels();
        StartWebCam();

        if (m_PermissionDialog != null)
            m_PermissionDialog.SetActive(false);
    }

    void StartWebCam()
    {
        var devices = WebCamTexture.devices;
        if (devices == null || devices.Length == 0)
        {
            m_StatusText.text = "Error: No camera found on this device!";
            return;
        }

        // Pick front or back camera
        m_ActiveCameraDeviceIndex = 0;
        for (int i = 0; i < devices.Length; i++)
        {
            if (preferFrontCamera && devices[i].isFrontFacing)
            {
                m_ActiveCameraDeviceIndex = i;
                break;
            }
            if (!preferFrontCamera && !devices[i].isFrontFacing)
            {
                m_ActiveCameraDeviceIndex = i;
                break;
            }
        }

        if (m_WebCamTexture != null)
        {
            m_WebCamTexture.Stop();
            Destroy(m_WebCamTexture);
        }

        var dev = devices[m_ActiveCameraDeviceIndex];
        m_WebCamTexture = new WebCamTexture(dev.name, requestedWidth, requestedHeight, requestedFPS);
        m_WebCamTexture.Play();

        m_CameraFeedRawImage.texture = m_WebCamTexture;
        m_CameraFeedRawImage.color = Color.white;

        m_IsRunning = true;
        RunTrackingLoop();
    }

    public void SwitchCamera()
    {
        var devices = WebCamTexture.devices;
        if (devices == null || devices.Length <= 1) return;

        m_ActiveCameraDeviceIndex = (m_ActiveCameraDeviceIndex + 1) % devices.Length;
        preferFrontCamera = devices[m_ActiveCameraDeviceIndex].isFrontFacing;
        StartWebCam();
    }

    public void CycleCameraRotation()
    {
        manualRotationAngle = (manualRotationAngle + 90) % 360;
        UpdateCameraFeedTransform();
        m_StatusText.text = $"Sensor Angle: {manualRotationAngle}° (Manual Override)";
    }

    void UpdateCameraFeedTransform()
    {
        if (m_WebCamTexture == null || m_WebCamTexture.width < 16) return;

        int videoAngle = m_WebCamTexture.videoRotationAngle;
        int totalAngle = (videoAngle + manualRotationAngle) % 360;
        bool isMirrored = m_WebCamTexture.videoVerticallyMirrored;

        // On mobile, if front camera, mirror horizontally so it feels natural
        bool isFront = WebCamTexture.devices[m_ActiveCameraDeviceIndex].isFrontFacing;

        m_CameraFeedRect.localEulerAngles = new Vector3(0, 0, -totalAngle);

        float aspect = (float)m_WebCamTexture.width / Mathf.Max(1, m_WebCamTexture.height);
        if (totalAngle == 90 || totalAngle == 270)
        {
            aspect = (float)m_WebCamTexture.height / Mathf.Max(1, m_WebCamTexture.width);
        }
        m_CameraFitter.aspectRatio = aspect;

        float scaleX = isFront ? -1f : 1f;
        float scaleY = isMirrored ? -1f : 1f;
        m_CameraFeedRect.localScale = new Vector3(scaleX, scaleY, 1f);
    }

    #endregion

    #region MediaPipe Model Initialization

    void InitializeModels()
    {
        if (m_ModelsLoaded) return;

        // Auto-assign from Assets/Resources if null
        if (handDetectorAsset == null)
            handDetectorAsset = Resources.Load<ModelAsset>("Models/hand_detector");
        if (handLandmarkerAsset == null)
            handLandmarkerAsset = Resources.Load<ModelAsset>("Models/hand_landmarks_detector");
        if (handAnchorsCSV == null)
            handAnchorsCSV = Resources.Load<TextAsset>("Data/hand_anchors");

#if UNITY_EDITOR
        if (handDetectorAsset == null)
            handDetectorAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/BlazeDetection/Models/hand_detector.onnx");
        if (handLandmarkerAsset == null)
            handLandmarkerAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/BlazeDetection/Models/hand_landmarks_detector.onnx");
        if (handAnchorsCSV == null)
            handAnchorsCSV = UnityEditor.AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/BlazeDetection/Data/hand_anchors.csv");
#endif

        if (handDetectorAsset == null || handLandmarkerAsset == null || handAnchorsCSV == null)
        {
            m_StatusText.text = "Error: MediaPipe Sentis models or anchors not found!";
            Debug.LogError("Sentis Models missing! Ensure hand_detector, hand_landmarks_detector, and hand_anchors are in Resources.");
            return;
        }

        m_Anchors = BlazeUtils.LoadAnchors(handAnchorsCSV.text, k_NumAnchors);

        // 1. Hand Detector with NMS
        var detModel = ModelLoader.Load(handDetectorAsset);
        var graph = new FunctionalGraph();
        var input = graph.AddInput(detModel, 0);
        var outputs = Functional.Forward(detModel, input);
        var anchorsData = new float[k_NumAnchors * 4];
        Buffer.BlockCopy(m_Anchors, 0, anchorsData, 0, anchorsData.Length * sizeof(float));
        var anchors = Functional.Constant(new TensorShape(k_NumAnchors, 4), anchorsData);
        var filtered = BlazeUtils.NMSFiltering(outputs[0], outputs[1], anchors, k_DetectorInputSize, handIouThreshold, handScoreThreshold);
        detModel = graph.Compile(filtered.Item1, filtered.Item2, filtered.Item3);

        m_DetectorWorker = new Worker(detModel, BackendType.GPUCompute);
        m_DetectorInputTensor = new Tensor<float>(new TensorShape(1, k_DetectorInputSize, k_DetectorInputSize, 3));

        // 2. Hand 3D Landmarker
        var landModel = ModelLoader.Load(handLandmarkerAsset);
        m_LandmarkerWorker = new Worker(landModel, BackendType.GPUCompute);
        m_LandmarkerInputTensor = new Tensor<float>(new TensorShape(1, k_LandmarkerInputSize, k_LandmarkerInputSize, 3));

        m_ModelsLoaded = true;
    }

    #endregion

    #region Hand Inference & Gesture Recognition Loop

    async void RunTrackingLoop()
    {
        while (m_IsRunning)
        {
            try
            {
                if (m_WebCamTexture != null && m_WebCamTexture.isPlaying && m_WebCamTexture.width > 32)
                {
                    await ProcessHandTracking(m_WebCamTexture);
                }
                await Awaitable.NextFrameAsync();
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                await Awaitable.NextFrameAsync();
            }
        }
    }

    async Awaitable ProcessHandTracking(Texture texture)
    {
        int sensorAngle = (m_WebCamTexture.videoRotationAngle + manualRotationAngle) % 360;
        float thetaRot = sensorAngle * Mathf.Deg2Rad;

        // Build affine sampling transformation with rotation compensation
        float size = Mathf.Max(texture.width, texture.height);
        float scale = size / (float)k_DetectorInputSize;
        float2 texCenter = 0.5f * new float2(texture.width, texture.height);
        float2 detOrigin = 0.5f * new float2(k_DetectorInputSize, k_DetectorInputSize);

        var M = BlazeUtils.mul(
            BlazeUtils.TranslationMatrix(texCenter),
            BlazeUtils.mul(
                BlazeUtils.RotationMatrix(thetaRot),
                BlazeUtils.mul(
                    BlazeUtils.ScaleMatrix(new float2(scale, -scale)),
                    BlazeUtils.TranslationMatrix(-detOrigin)
                )
            )
        );

        BlazeUtils.SampleImageAffine(texture, m_DetectorInputTensor, M);
        m_DetectorWorker.Schedule(m_DetectorInputTensor);

        var readIdx = (m_DetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
        var readScores = (m_DetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
        var readBoxes = (m_DetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

        using var outputIdx = await readIdx;
        using var outputScores = await readScores;
        using var outputBoxes = await readBoxes;

        int numDetections = outputScores.shape[1];
        int activeHands = 0;
        string topGesture = "None";

        for (int h = 0; h < k_MaxHands; h++)
        {
            if (h < numDetections && outputScores[0, h] >= handScoreThreshold && activeHands < maxHands)
            {
                float score = outputScores[0, h];
                var boxCenter = new float2(outputBoxes[0, h, 0], outputBoxes[0, h, 1]);
                float boxSize = Mathf.Max(outputBoxes[0, h, 2], outputBoxes[0, h, 3]);

                var kp0 = new float2(outputBoxes[0, h, 4 + 2 * 0 + 0], outputBoxes[0, h, 4 + 2 * 0 + 1]);
                var kp2 = new float2(outputBoxes[0, h, 4 + 2 * 2 + 0], outputBoxes[0, h, 4 + 2 * 2 + 1]);
                var delta = kp2 - kp0;
                float rotAngle = 0.5f * Mathf.PI - math.atan2(delta.y, delta.x);
                var upDir = delta / Mathf.Max(0.001f, math.length(delta));

                boxCenter += 0.5f * boxSize * upDir;
                boxSize *= 2.6f;

                float2 landOrigin = 0.5f * new float2(k_LandmarkerInputSize, k_LandmarkerInputSize);
                float landScale = boxSize / (float)k_LandmarkerInputSize;

                var M2 = BlazeUtils.mul(M, BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.mul(
                    BlazeUtils.TranslationMatrix(boxCenter),
                    BlazeUtils.ScaleMatrix(new float2(landScale, -landScale))),
                    BlazeUtils.RotationMatrix(rotAngle)),
                    BlazeUtils.TranslationMatrix(-landOrigin)));

                BlazeUtils.SampleImageAffine(texture, m_LandmarkerInputTensor, M2);
                m_LandmarkerWorker.Schedule(m_LandmarkerInputTensor);

                var readLandmarks = (m_LandmarkerWorker.PeekOutput("Identity") as Tensor<float>).ReadbackAndCloneAsync();
                using var landmarks = await readLandmarks;

                // Project landmarks into Screen & Canvas Space
                var canvasPositions = new Vector2[k_NumLandmarks];
                var raw3DLandmarks = new Vector3[k_NumLandmarks];
                var canvasRect = m_Canvas.GetComponent<RectTransform>();
                float canvasW = canvasRect.rect.width;
                float canvasH = canvasRect.rect.height;

                for (int i = 0; i < k_NumLandmarks; i++)
                {
                    float2 imgPos = BlazeUtils.mul(M2, new float2(landmarks[3 * i + 0], landmarks[3 * i + 1]));
                    raw3DLandmarks[i] = new Vector3(landmarks[3 * i + 0], landmarks[3 * i + 1], landmarks[3 * i + 2]);

                    // Map texture pixel coords [0..W, 0..H] into normalized [0..1]
                    float u = imgPos.x / (float)texture.width;
                    float v = 1f - (imgPos.y / (float)texture.height);

                    // Compensate for mirroring if front camera
                    bool isFront = WebCamTexture.devices[m_ActiveCameraDeviceIndex].isFrontFacing;
                    if (isFront) u = 1f - u;

                    // Convert to Canvas centered anchored position
                    canvasPositions[i] = new Vector2((u - 0.5f) * canvasW, (v - 0.5f) * canvasH);
                }

                // Recognize Gesture
                string gesture = ClassifyGesture(raw3DLandmarks);
                topGesture = gesture;

                // Render Canvas Visuals for this Hand
                RenderHandOnCanvas(h, canvasPositions, gesture, score);
                activeHands++;
            }
            else
            {
                SetHandVisible(h, false);
            }
        }

        // Update Top HUD
        m_StatusText.text = $"FPS: {m_CurrentFps} | Hands: {activeHands} | Resolution: {texture.width}x{texture.height}";
        m_GestureSummaryText.text = activeHands > 0 ? $"Gesture: {topGesture}" : "Gesture: Show your hand";
    }

    void RenderHandOnCanvas(int handIndex, Vector2[] landmarks, string gesture, float confidence)
    {
        SetHandVisible(handIndex, true);

        // Update Joints
        for (int j = 0; j < k_NumLandmarks; j++)
        {
            var jointRt = m_JointMarkers[handIndex][j];
            jointRt.anchoredPosition = landmarks[j];
        }

        // Update Bones
        for (int b = 0; b < HandBones.Length; b++)
        {
            var (startIdx, endIdx) = HandBones[b];
            Vector2 p1 = landmarks[startIdx];
            Vector2 p2 = landmarks[endIdx];

            var boneRt = m_BoneLines[handIndex][b];
            Vector2 dir = p2 - p1;
            float dist = dir.magnitude;

            boneRt.anchoredPosition = p1 + 0.5f * dir;
            boneRt.sizeDelta = new Vector2(dist, 5f);
            float angle = Mathf.Atan2(dir.y, dir.x) * Mathf.Rad2Deg;
            boneRt.localEulerAngles = new Vector3(0, 0, angle);
        }

        // Position Gesture Badge near the wrist/palm
        Vector2 wristPos = landmarks[0];
        m_GestureBadgeRects[handIndex].anchoredPosition = wristPos + new Vector2(0f, -65f);
        m_GestureBadges[handIndex].text = $"{gesture} ({Mathf.RoundToInt(confidence * 100)}%)";
    }

    void SetHandVisible(int handIndex, bool visible)
    {
        if (m_HandRoots[handIndex] == null) return;

        for (int j = 0; j < k_NumLandmarks; j++)
        {
            if (m_JointMarkers[handIndex][j] != null)
                m_JointMarkers[handIndex][j].gameObject.SetActive(visible);
        }
        for (int b = 0; b < HandBones.Length; b++)
        {
            if (m_BoneLines[handIndex][b] != null)
                m_BoneLines[handIndex][b].gameObject.SetActive(visible);
        }
        if (m_GestureBadgeRects[handIndex] != null)
        {
            m_GestureBadgeRects[handIndex].gameObject.SetActive(visible);
        }
    }

    #endregion

    #region Gesture Classifier

    /// <summary>
    /// Analyzes the 21 MediaPipe hand landmarks and classifies the gesture.
    /// Extremely fast and robust against camera distance and rotation.
    /// </summary>
    string ClassifyGesture(Vector3[] lm)
    {
        if (lm == null || lm.Length < 21) return "Unknown";

        // Check finger extensions
        // Index: 8 (Tip) vs 6 (PIP)
        // Middle: 12 (Tip) vs 10 (PIP)
        // Ring: 16 (Tip) vs 14 (PIP)
        // Pinky: 20 (Tip) vs 18 (PIP)
        Vector3 wrist = lm[0];

        bool indexOpen = Vector3.Distance(lm[8], wrist) > Vector3.Distance(lm[6], wrist);
        bool middleOpen = Vector3.Distance(lm[12], wrist) > Vector3.Distance(lm[10], wrist);
        bool ringOpen = Vector3.Distance(lm[16], wrist) > Vector3.Distance(lm[14], wrist);
        bool pinkyOpen = Vector3.Distance(lm[20], wrist) > Vector3.Distance(lm[18], wrist);

        // Thumb: 4 (Tip) vs 2 (MCP) relative to pinky base (17)
        bool thumbOpen = Vector3.Distance(lm[4], lm[17]) > Vector3.Distance(lm[3], lm[17]);

        // Pinch check (Thumb tip to Index tip distance)
        float pinchDistance = Vector3.Distance(lm[4], lm[8]);
        float palmScale = Vector3.Distance(lm[0], lm[9]); // Wrist to Middle MCP as reference
        bool isPinch = pinchDistance < (palmScale * 0.32f);

        // Classification Rules:
        if (isPinch && middleOpen && ringOpen && pinkyOpen)
        {
            return "👌 Pinch / OK";
        }
        if (thumbOpen && indexOpen && middleOpen && ringOpen && pinkyOpen)
        {
            return "🖐️ Open Palm";
        }
        if (!thumbOpen && !indexOpen && !middleOpen && !ringOpen && !pinkyOpen)
        {
            return "✊ Fist";
        }
        if (!thumbOpen && indexOpen && middleOpen && !ringOpen && !pinkyOpen)
        {
            return "✌️ Peace / V";
        }
        if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen)
        {
            return "👉 Pointing";
        }
        if (thumbOpen && !indexOpen && !middleOpen && !ringOpen && !pinkyOpen)
        {
            return "👍 Thumbs Up";
        }
        if (thumbOpen && indexOpen && !middleOpen && !ringOpen && pinkyOpen)
        {
            return "🤟 Rock On";
        }

        return "✋ Hand Detected";
    }

    #endregion
}
