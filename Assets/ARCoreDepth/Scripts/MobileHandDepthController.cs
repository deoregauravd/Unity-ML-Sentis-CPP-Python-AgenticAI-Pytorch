using System;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Mathematics;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

/// <summary>
/// Main mobile application controller that coordinates:
/// 1. Camera permissions & desktop webcam fallback
/// 2. ARCore depth mapping via ARFoundation AROcclusionManager
/// 3. Real-time Sentis BlazeHand 21-landmark tracking
/// 4. Dynamic UI overlays: fullscreen camera background, 21-joint skeleton, depth map preview
/// 5. Depth-to-hand fusion: computing physical distance in meters to the user's hand
/// </summary>
public class MobileHandDepthController : MonoBehaviour
{
    [Header("Managers")]
    public CameraPermissionHelper permissionHelper;
    public ARCoreDepthManager depthManager;
    public ARCameraManager arCameraManager;

    [Header("BlazeHand Models & Anchors")]
    public ModelAsset handDetectorAsset;
    public ModelAsset handLandmarkerAsset;
    public TextAsset handAnchorsCSV;

    [Header("UI References")]
    public Text statusText;
    public Text distanceText;
    public RawImage depthPreviewUI;
    public Button toggleDepthButton;
    public Button toggleHandButton;

    [Header("Camera Feed Background (Editor & Standalone)")]
    public RawImage cameraBackgroundUI;

    [Header("Orientation & Mirroring")]
    [Tooltip("Mirror webcam feed horizontally so it acts like a selfie mirror (default: true for PC, false for Android back camera)")]
    public bool mirrorHorizontally = true;

    [Tooltip("Flip webcam feed vertically if required by specific camera drivers (default: false)")]
    public bool flipVertically = false;

    [Tooltip("Camera sensor rotation angle offset (default 0; mobile auto-uses WebCamTexture.videoRotationAngle)")]
    public int cameraRotationAngle = 0;

    [Header("Multi-Hand Tracking")]
    [Tooltip("Maximum hands to detect and track concurrently")]
    [Range(1, 2)]
    public int maxHands = 2;
    [Range(0.01f, 1f)] public float handScoreThreshold = 0.4f;
    [Range(0.01f, 1f)] public float handIouThreshold = 0.3f;

    [Header("3D VR Holographic Hands")]
    public VRHolographicHand vrHandVisualizer;          // Hand 0 (Left/Primary)
    public VRHolographicHand vrHandVisualizerSecondary; // Hand 1 (Right/Secondary)
    public VRHolographicHand[] vrHandVisualizers;
    public bool showVRHand = true;
    public bool showSkeleton = true;
    [Tooltip("If true, shows full uncropped/unzoomed camera frame (FitInParent). If false, fills screen (EnvelopeParent).")]
    public bool fitCameraFeed = true;
    public Color hand0Color = new Color(0.1f, 0.85f, 1f, 0.85f);  // Cyan
    public Color hand1Color = new Color(0.85f, 0.3f, 1f, 0.85f);  // Neon Magenta / Violet

    [Header("Visual Styling (Skeleton Mode)")]
    public Color jointColor = new Color(0.2f, 0.85f, 1f, 1f);
    public Color boneColor = new Color(1f, 0.9f, 0.2f, 0.9f);

    Text[] m_UIDepthBadges = new Text[k_MaxHands];
    RectTransform[] m_UIDepthBadgeRects = new RectTransform[k_MaxHands];

    // Hand Skeleton Constants
    const int k_MaxHands = 2;
    const int k_NumAnchors = 2016;
    const int k_DetectorInputSize = 192;
    const int k_LandmarkerInputSize = 224;
    const int k_NumKeypoints = 21;

    static readonly (int, int)[] HandBones = new (int, int)[]
    {
        (0, 1), (1, 2), (2, 3), (3, 4),        // Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        // Index
        (5, 9), (9, 10), (10, 11), (11, 12),   // Middle
        (9, 13), (13, 14), (14, 15), (15, 16), // Ring
        (13, 17), (17, 18), (18, 19), (19, 20),// Pinky
        (0, 17)                                // Palm Base
    };

    // State
    float[,] m_Anchors;
    Worker m_DetectorWorker;
    Worker m_LandmarkerWorker;
    Tensor<float> m_DetectorInput;
    Tensor<float> m_LandmarkerInput;

    WebCamTexture m_EditorWebCam;
    Texture2D m_CameraFrameTexture;

    Material m_LineMaterial;
    GameObject[] m_HandVisualsRoots = new GameObject[k_MaxHands];
    LineRenderer[][] m_JointRenderers = new LineRenderer[k_MaxHands][];
    LineRenderer[][] m_BoneRenderers = new LineRenderer[k_MaxHands][];

    // 2D UI Skeleton Overlay (Renders directly on top of camera feed)
    RectTransform[] m_UIHandRoots = new RectTransform[k_MaxHands];
    RectTransform[][] m_UIJoints = new RectTransform[k_MaxHands][];
    RectTransform[][] m_UIBones = new RectTransform[k_MaxHands][];

    bool m_PermissionGranted;
    bool m_IsRunning;
    bool m_ShowDepthMap = true;
    bool m_ShowHandTracking = true;

    Camera m_MainCamera;

    void Awake()
    {
        m_MainCamera = Camera.main;

        // Auto-resolve managers if not set in Inspector
        if (permissionHelper == null)
            permissionHelper = GetComponent<CameraPermissionHelper>();
        if (depthManager == null)
            depthManager = FindAnyObjectByType<ARCoreDepthManager>();
        if (arCameraManager == null)
            arCameraManager = FindAnyObjectByType<ARCameraManager>();

        // Auto-resolve UI references if not set in Inspector
        if (statusText == null)
            statusText = GameObject.Find("StatusText")?.GetComponent<Text>();
        if (distanceText == null)
            distanceText = GameObject.Find("DistanceText")?.GetComponent<Text>();
        if (depthPreviewUI == null)
            depthPreviewUI = GameObject.Find("DepthPreviewUI")?.GetComponent<RawImage>();
        if (toggleDepthButton == null)
            toggleDepthButton = GameObject.Find("ToggleDepthButton")?.GetComponent<Button>();
        if (toggleHandButton == null)
            toggleHandButton = GameObject.Find("ToggleHandButton")?.GetComponent<Button>();

        if (permissionHelper != null && permissionHelper.statusText == null)
            permissionHelper.statusText = statusText;

        if (depthManager != null && depthPreviewUI != null && depthManager.depthMapDisplay == null)
            depthManager.depthMapDisplay = depthPreviewUI;

        // Auto-resolve or create 3D VR Holographic Hands (for both hands)
        if (vrHandVisualizers == null || vrHandVisualizers.Length < k_MaxHands)
            vrHandVisualizers = new VRHolographicHand[k_MaxHands];

        // Hand 0 (Primary)
        if (vrHandVisualizer == null)
            vrHandVisualizer = transform.Find("VRHolographicHand_0")?.GetComponent<VRHolographicHand>()
                            ?? transform.Find("VRHolographicHand")?.GetComponent<VRHolographicHand>()
                            ?? GetComponentInChildren<VRHolographicHand>();
        if (vrHandVisualizer == null)
        {
            var vrObj0 = new GameObject("VRHolographicHand_0", typeof(VRHolographicHand));
            vrObj0.transform.SetParent(transform, false);
            vrHandVisualizer = vrObj0.GetComponent<VRHolographicHand>();
        }
        vrHandVisualizers[0] = vrHandVisualizer;
        vrHandVisualizers[0].SetThemeColor(hand0Color, new Color(0.4f, 0.95f, 1f, 1f));

        // Hand 1 (Secondary)
        if (vrHandVisualizerSecondary == null)
            vrHandVisualizerSecondary = transform.Find("VRHolographicHand_1")?.GetComponent<VRHolographicHand>();
        if (vrHandVisualizerSecondary == null)
        {
            var vrObj1 = new GameObject("VRHolographicHand_1", typeof(VRHolographicHand));
            vrObj1.transform.SetParent(transform, false);
            vrHandVisualizerSecondary = vrObj1.GetComponent<VRHolographicHand>();
        }
        vrHandVisualizers[1] = vrHandVisualizerSecondary;
        vrHandVisualizers[1].SetThemeColor(hand1Color, new Color(1f, 0.5f, 1f, 1f));

        showSkeleton = true;
        fitCameraFeed = true;

#if !UNITY_EDITOR
        // On mobile with AR back camera, do not mirror by default so real hands match real-world positions
        mirrorHorizontally = false;
#endif

        // Auto-format Canvas and UI positioning
        ConfigureCanvasAndUI();

        CreateLineMaterial();
        CreateHandVisualizers();
        CreateUIHandVisualizers();
        SetupUIListeners();
    }

    GameObject m_StartCard;

    void Start()
    {
        if (permissionHelper != null)
        {
            permissionHelper.OnPermissionGranted += OnPermissionGranted;
            permissionHelper.OnPermissionDenied += OnPermissionDenied;
        }

        CreateStartCard();
    }

    void CreateStartCard()
    {
        var canvas = FindAnyObjectByType<Canvas>();
        if (canvas == null)
        {
            OnPermissionGranted();
            return;
        }

        m_StartCard = new GameObject("StartPermissionCard", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        m_StartCard.transform.SetParent(canvas.transform, false);
        var rt = m_StartCard.GetComponent<RectTransform>();
        rt.anchorMin = new Vector2(0.5f, 0.5f);
        rt.anchorMax = new Vector2(0.5f, 0.5f);
        rt.pivot = new Vector2(0.5f, 0.5f);
        rt.sizeDelta = new Vector2(680f, 440f);
        rt.anchoredPosition = Vector2.zero;

        var img = m_StartCard.GetComponent<Image>();
        img.color = new Color(0.08f, 0.12f, 0.18f, 0.95f);
        var outline = m_StartCard.AddComponent<Outline>();
        outline.effectColor = new Color(0f, 0.9f, 1f, 0.9f);
        outline.effectDistance = new Vector2(3, -3);

        // Title
        var titleGo = new GameObject("Title", typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        titleGo.transform.SetParent(m_StartCard.transform, false);
        var tRt = titleGo.GetComponent<RectTransform>();
        tRt.anchorMin = new Vector2(0f, 0.7f);
        tRt.anchorMax = new Vector2(1f, 1f);
        tRt.anchoredPosition = new Vector2(0f, -20f);
        var titleTxt = titleGo.GetComponent<Text>();
        titleTxt.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
        titleTxt.fontSize = 32;
        titleTxt.fontStyle = FontStyle.Bold;
        titleTxt.alignment = TextAnchor.MiddleCenter;
        titleTxt.color = new Color(0f, 0.95f, 1f, 1f);
        titleTxt.text = "MediaPipe Hand Gestures";

        // Message
        var msgGo = new GameObject("Message", typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        msgGo.transform.SetParent(m_StartCard.transform, false);
        var mRt = msgGo.GetComponent<RectTransform>();
        mRt.anchorMin = new Vector2(0.08f, 0.35f);
        mRt.anchorMax = new Vector2(0.92f, 0.72f);
        mRt.anchoredPosition = Vector2.zero;
        var msgTxt = msgGo.GetComponent<Text>();
        msgTxt.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
        msgTxt.fontSize = 22;
        msgTxt.alignment = TextAnchor.MiddleCenter;
        msgTxt.color = new Color(0.9f, 0.9f, 0.95f, 1f);
        msgTxt.text = "Track hands and recognize gestures in real-time.\n\nTap below to enable camera and begin tracking.";

        // Start Button
        var btnGo = new GameObject("StartButton", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(Button), typeof(Outline));
        btnGo.transform.SetParent(m_StartCard.transform, false);
        var bRt = btnGo.GetComponent<RectTransform>();
        bRt.anchorMin = new Vector2(0.5f, 0.18f);
        bRt.anchorMax = new Vector2(0.5f, 0.18f);
        bRt.pivot = new Vector2(0.5f, 0.5f);
        bRt.sizeDelta = new Vector2(380f, 75f);
        bRt.anchoredPosition = Vector2.zero;

        var bImg = btnGo.GetComponent<Image>();
        bImg.color = new Color(0.12f, 0.5f, 0.95f, 0.95f);
        var bOutline = btnGo.GetComponent<Outline>();
        bOutline.effectColor = new Color(0.3f, 0.85f, 1f, 0.9f);
        bOutline.effectDistance = new Vector2(2, -2);

        var btnTextGo = new GameObject("Text", typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        btnTextGo.transform.SetParent(btnGo.transform, false);
        var btRt = btnTextGo.GetComponent<RectTransform>();
        btRt.anchorMin = Vector2.zero;
        btRt.anchorMax = Vector2.one;
        btRt.sizeDelta = Vector2.zero;
        var bTxt = btnTextGo.GetComponent<Text>();
        bTxt.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
        bTxt.fontSize = 24;
        bTxt.fontStyle = FontStyle.Bold;
        bTxt.alignment = TextAnchor.MiddleCenter;
        bTxt.color = Color.white;
        bTxt.text = "Start Camera & Tracking";

        var btn = btnGo.GetComponent<Button>();
        btn.onClick.AddListener(() =>
        {
            if (m_StartCard != null) m_StartCard.SetActive(false);
            if (permissionHelper != null)
            {
                permissionHelper.StartCoroutine(permissionHelper.CheckAndRequestCameraPermission());
            }
            else
            {
                OnPermissionGranted();
            }
        });
    }

    ScreenOrientation m_LastOrientation;
    int m_LastScreenWidth;
    int m_LastScreenHeight;

    void Update()
    {
        if (Screen.width != m_LastScreenWidth || Screen.height != m_LastScreenHeight || Screen.orientation != m_LastOrientation)
        {
            m_LastScreenWidth = Screen.width;
            m_LastScreenHeight = Screen.height;
            m_LastOrientation = Screen.orientation;
            ConfigureCanvasAndUI();
            UpdateWebCamOrientation();
        }
        else
        {
            UpdateWebCamOrientation();
        }

        // Hotkeys to quickly toggle mirror mode (M), flip vertical (F), or VR Hologram Hand (H)
        if (Keyboard.current != null)
        {
            if (Keyboard.current.mKey.wasPressedThisFrame)
            {
                mirrorHorizontally = !mirrorHorizontally;
                UpdateWebCamOrientation();
                SetStatus($"Mirror Mode: {(mirrorHorizontally ? "ON" : "OFF")}");
            }
            if (Keyboard.current.fKey.wasPressedThisFrame)
            {
                flipVertically = !flipVertically;
                UpdateWebCamOrientation();
                SetStatus($"Flip Vertical: {(flipVertically ? "ON" : "OFF")}");
            }
            if (Keyboard.current.rKey.wasPressedThisFrame)
            {
                cameraRotationAngle = (cameraRotationAngle + 90) % 360;
                SetStatus($"Camera Rotation: {cameraRotationAngle}°");
            }
            if (Keyboard.current.hKey.wasPressedThisFrame)
            {
                // Cycle: VR Hand Only -> Both -> Skeleton Only
                if (showVRHand && !showSkeleton)
                {
                    showSkeleton = true; // Both
                }
                else if (showVRHand && showSkeleton)
                {
                    showVRHand = false; // Skeleton only
                }
                else
                {
                    showVRHand = true;
                    showSkeleton = false; // VR Hologram only
                }

                SetHandVisualsActive(m_IsRunning);
                SetStatus($"Display: {(showVRHand ? "VR Hologram " : "")}{(showSkeleton ? "+ Skeleton" : "")}");
            }
        }

        // Tap/click on screen to query metric depth using the new Input System
        if (Pointer.current != null && Pointer.current.press.wasPressedThisFrame)
        {
            Vector2 pointerPos = Pointer.current.position.ReadValue();
            if (depthManager != null && depthManager.TryGetDepthAtScreenPoint(pointerPos, out float depthMeters))
            {
                if (distanceText != null)
                    distanceText.text = $"Point Depth: {depthMeters:F2} m";
            }
        }
    }

    void ConfigureCanvasAndUI()
    {
        var canvas = FindAnyObjectByType<Canvas>();
        if (canvas == null) return;

#if UNITY_EDITOR
        // In Editor, use ScreenSpaceCamera so cameraBackgroundUI doesn't occlude 3D VR hands
        canvas.renderMode = RenderMode.ScreenSpaceCamera;
        canvas.worldCamera = m_MainCamera;
        canvas.planeDistance = 25f;
#else
        // On Mobile (Android/iOS), ALWAYS use ScreenSpaceOverlay!
        // ARCameraBackground handles rendering the camera feed in 3D background.
        // ScreenSpaceOverlay guarantees the UI occupies 100% of any phone screen!
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
#endif

        var scaler = canvas.GetComponent<CanvasScaler>();
        if (scaler != null)
        {
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            bool isPortrait = Screen.height > Screen.width;
            scaler.referenceResolution = isPortrait ? new Vector2(1080, 1920) : new Vector2(1920, 1080);
            scaler.matchWidthOrHeight = isPortrait ? 0f : 1f;
        }

        // Layout UI elements with responsive anchors:
        if (statusText != null)
        {
            var rt = statusText.rectTransform;
            rt.anchorMin = new Vector2(0f, 1f);
            rt.anchorMax = new Vector2(1f, 1f);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.anchoredPosition = new Vector2(0f, -30f);
            rt.sizeDelta = new Vector2(-40f, 60f);
            statusText.alignment = TextAnchor.MiddleCenter;
            statusText.fontSize = 24;
            statusText.color = Color.white;
        }

        if (distanceText != null)
        {
            var rt = distanceText.rectTransform;
            rt.anchorMin = new Vector2(0f, 1f);
            rt.anchorMax = new Vector2(1f, 1f);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.anchoredPosition = new Vector2(0f, -95f);
            rt.sizeDelta = new Vector2(-40f, 60f);
            distanceText.alignment = TextAnchor.MiddleCenter;
            distanceText.fontSize = 26;
            distanceText.color = new Color(1f, 0.9f, 0.2f, 1f);
        }

        if (depthPreviewUI != null)
        {
            var rt = depthPreviewUI.rectTransform;
            rt.anchorMin = new Vector2(1f, 1f);
            rt.anchorMax = new Vector2(1f, 1f);
            rt.pivot = new Vector2(1f, 1f);
            rt.anchoredPosition = new Vector2(-25f, -165f);
            rt.sizeDelta = new Vector2(180f, 240f);
        }

        if (toggleDepthButton != null)
        {
            var rt = toggleDepthButton.GetComponent<RectTransform>();
            if (rt != null)
            {
                rt.anchorMin = new Vector2(0f, 0f);
                rt.anchorMax = new Vector2(0f, 0f);
                rt.pivot = new Vector2(0f, 0f);
                rt.anchoredPosition = new Vector2(30f, 40f);
                rt.sizeDelta = new Vector2(220f, 80f);
            }
        }

        if (toggleHandButton != null)
        {
            var rt = toggleHandButton.GetComponent<RectTransform>();
            if (rt != null)
            {
                rt.anchorMin = new Vector2(1f, 0f);
                rt.anchorMax = new Vector2(1f, 0f);
                rt.pivot = new Vector2(1f, 0f);
                rt.anchoredPosition = new Vector2(-30f, 40f);
                rt.sizeDelta = new Vector2(220f, 80f);
            }
        }
    }

    void SetupUIListeners()
    {
        if (toggleDepthButton != null)
        {
            var btnText = toggleDepthButton.GetComponentInChildren<Text>();
            if (btnText != null) btnText.text = "Camera: Fit";

            toggleDepthButton.onClick.AddListener(() =>
            {
                fitCameraFeed = !fitCameraFeed;
                UpdateWebCamOrientation();
                if (btnText != null)
                    btnText.text = fitCameraFeed ? "Camera: Fit" : "Camera: Fill";
                SetStatus($"Camera View: {(fitCameraFeed ? "Fit (Full View Unzoomed)" : "Fill (Zoomed)")}");
            });
        }

        if (toggleHandButton != null)
        {
            var btnText = toggleHandButton.GetComponentInChildren<Text>();
            if (btnText != null) btnText.text = "Skeleton: ON";

            toggleHandButton.onClick.AddListener(() =>
            {
                showSkeleton = !showSkeleton;
                SetHandVisualsActive(m_IsRunning);
                if (btnText != null)
                    btnText.text = showSkeleton ? "Skeleton: ON" : "Skeleton: OFF";
                SetStatus($"Skeleton: {(showSkeleton ? "ON" : "OFF")}");
            });
        }

        if (statusText != null)
        {
            var btn = statusText.GetComponent<Button>();
            if (btn == null)
            {
                btn = statusText.gameObject.AddComponent<Button>();
            }
            btn.onClick.AddListener(() =>
            {
                cameraRotationAngle = (cameraRotationAngle + 90) % 360;
                SetStatus($"Camera Sensor Orientation: {cameraRotationAngle}° (Tap to rotate)");
            });
        }
    }

    void OnPermissionGranted()
    {
        m_PermissionGranted = true;
        SetStatus("Initializing AR Depth & Hand Models...");

        InitializeBlazeHand();
        StartProcessing();
    }

    void OnPermissionDenied()
    {
        SetStatus("Camera permission was denied. Cannot start tracking.");
    }

    void InitializeBlazeHand()
    {
#if UNITY_EDITOR
        if (handAnchorsCSV == null)
            handAnchorsCSV = UnityEditor.AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/BlazeDetection/Data/hand_anchors.csv");
        if (handDetectorAsset == null)
            handDetectorAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/BlazeDetection/Models/hand_detector.onnx");
        if (handLandmarkerAsset == null)
            handLandmarkerAsset = UnityEditor.AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/BlazeDetection/Models/hand_landmarks_detector.onnx");
#endif
        if (handAnchorsCSV == null)
            handAnchorsCSV = Resources.Load<TextAsset>("Data/hand_anchors");
        if (handDetectorAsset == null)
            handDetectorAsset = Resources.Load<ModelAsset>("Models/hand_detector");
        if (handLandmarkerAsset == null)
            handLandmarkerAsset = Resources.Load<ModelAsset>("Models/hand_landmarks_detector");

        if (handAnchorsCSV != null)
        {
            m_Anchors = BlazeUtils.LoadAnchors(handAnchorsCSV.text, k_NumAnchors);
        }
        else
        {
            Debug.LogError("[MobileHandDepthController] handAnchorsCSV not found!");
        }

        if (handDetectorAsset != null && handAnchorsCSV != null)
        {
            var detectorModel = ModelLoader.Load(handDetectorAsset);
            var graph = new FunctionalGraph();
            var input = graph.AddInput(detectorModel, 0);
            var outputs = Functional.Forward(detectorModel, input);

            var anchorsData = new float[k_NumAnchors * 4];
            Buffer.BlockCopy(m_Anchors, 0, anchorsData, 0, anchorsData.Length * sizeof(float));
            var anchors = Functional.Constant(new TensorShape(k_NumAnchors, 4), anchorsData);

            // NMS filtering detects both hands concurrently with non-maximum suppression
            var filtered = BlazeUtils.NMSFiltering(outputs[0], outputs[1], anchors, k_DetectorInputSize, handIouThreshold, handScoreThreshold);
            detectorModel = graph.Compile(filtered.Item1, filtered.Item2, filtered.Item3);

            m_DetectorWorker = new Worker(detectorModel, BackendType.GPUCompute);
            m_DetectorInput = new Tensor<float>(new TensorShape(1, k_DetectorInputSize, k_DetectorInputSize, 3));
        }
        else
        {
            Debug.LogError("[MobileHandDepthController] handDetectorAsset or handAnchorsCSV not found!");
        }

        if (handLandmarkerAsset != null)
        {
            var landmarkerModel = ModelLoader.Load(handLandmarkerAsset);
            m_LandmarkerWorker = new Worker(landmarkerModel, BackendType.GPUCompute);
            m_LandmarkerInput = new Tensor<float>(new TensorShape(1, k_LandmarkerInputSize, k_LandmarkerInputSize, 3));
        }
        else
        {
            Debug.LogError("[MobileHandDepthController] handLandmarkerAsset not found!");
        }
    }

    void StartProcessing()
    {
        if (WebCamTexture.devices != null && WebCamTexture.devices.Length > 0)
        {
            string devName = WebCamTexture.devices[0].name;
            for (int i = 0; i < WebCamTexture.devices.Length; i++)
            {
#if UNITY_ANDROID || UNITY_IOS
                if (!WebCamTexture.devices[i].isFrontFacing)
                {
                    devName = WebCamTexture.devices[i].name;
                    break;
                }
#endif
            }
            m_EditorWebCam = new WebCamTexture(devName, 1280, 720, 30);
            m_EditorWebCam.Play();
            SetupCameraBackground();
            SetStatus($"Camera Active ({devName}). Show your hands!");
        }
        else
        {
            SetStatus("No camera device found on device.");
        }

        m_IsRunning = true;
        RunInferenceLoop();
    }

    void SetupCameraBackground()
    {
        var canvas = FindAnyObjectByType<Canvas>();
        if (canvas == null) return;

        if (cameraBackgroundUI == null)
        {
            var bgObj = GameObject.Find("CameraBackgroundUI");
            if (bgObj != null)
            {
                cameraBackgroundUI = bgObj.GetComponent<RawImage>();
            }
            else
            {
                var newBg = new GameObject("CameraBackgroundUI", typeof(RectTransform), typeof(CanvasRenderer), typeof(RawImage));
                newBg.transform.SetParent(canvas.transform, false);
                newBg.transform.SetAsFirstSibling();
                cameraBackgroundUI = newBg.GetComponent<RawImage>();
            }
        }

        if (cameraBackgroundUI != null)
        {
            var fitter = cameraBackgroundUI.GetComponent<AspectRatioFitter>();
            if (fitter != null) Destroy(fitter);

            if (m_EditorWebCam != null)
            {
                cameraBackgroundUI.texture = m_EditorWebCam;
                cameraBackgroundUI.color = Color.white;
                cameraBackgroundUI.gameObject.SetActive(true);
                UpdateWebCamOrientation();
            }
        }
    }

    public int GetTotalVideoAngle()
    {
        int baseAngle = (m_EditorWebCam != null) ? m_EditorWebCam.videoRotationAngle : 0;
#if (UNITY_ANDROID || UNITY_IOS) && !UNITY_EDITOR
        if (m_EditorWebCam == null && Screen.height > Screen.width)
        {
            baseAngle = 90;
        }
#endif
        return (baseAngle + cameraRotationAngle) % 360;
    }

    void UpdateWebCamOrientation()
    {
        if (cameraBackgroundUI == null) return;

        // Ensure AspectRatioFitter is removed as it corrupts rotated RectTransforms
        var fitter = cameraBackgroundUI.GetComponent<AspectRatioFitter>();
        if (fitter != null) Destroy(fitter);

        int texW = (m_EditorWebCam != null && m_EditorWebCam.width > 32) ? m_EditorWebCam.width : 1280;
        int texH = (m_EditorWebCam != null && m_EditorWebCam.height > 32) ? m_EditorWebCam.height : 720;

        var canvas = FindAnyObjectByType<Canvas>();
        float screenW = Screen.width;
        float screenH = Screen.height;
        if (canvas != null)
        {
            var canvasRt = canvas.GetComponent<RectTransform>();
            if (canvasRt != null)
            {
                screenW = canvasRt.rect.width;
                screenH = canvasRt.rect.height;
            }
        }

        int videoAngle = GetTotalVideoAngle();
        bool isRotated90 = (videoAngle == 90 || videoAngle == 270);

        // Aspect ratio of the video feed as perceived upright by the user
        float visualAspect = isRotated90 ? ((float)texH / texW) : ((float)texW / texH);
        float screenAspect = screenW / Mathf.Max(1f, screenH);

        float visualW, visualH;
        if (fitCameraFeed)
        {
            // FIT MODE: Entire camera frame is visible in upright portrait without cropping
            if (screenAspect > visualAspect)
            {
                visualH = screenH;
                visualW = screenH * visualAspect;
            }
            else
            {
                visualW = screenW;
                visualH = screenW / visualAspect;
            }
        }
        else
        {
            // FILL MODE: Covers 100% of the screen with zero black bars / letterboxing
            if (screenAspect > visualAspect)
            {
                visualW = screenW;
                visualH = screenW / visualAspect;
            }
            else
            {
                visualH = screenH;
                visualW = screenH * visualAspect;
            }
        }

        // When rotated 90 or 270 degrees, local X corresponds to screen Y and local Y to screen X
        float localW = isRotated90 ? visualH : visualW;
        float localH = isRotated90 ? visualW : visualH;

        var rect = cameraBackgroundUI.rectTransform;
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = Vector2.zero;
        rect.sizeDelta = new Vector2(localW, localH);
        rect.localEulerAngles = new Vector3(0, 0, -videoAngle);

        bool isMirrored = m_EditorWebCam != null && m_EditorWebCam.videoVerticallyMirrored;
        cameraBackgroundUI.transform.localScale = new Vector3(
            mirrorHorizontally ? -1f : 1f,
            (isMirrored ^ flipVertically) ? -1f : 1f,
            1f
        );

        cameraBackgroundUI.uvRect = new Rect(0, 0, 1, 1);
    }

    async void RunInferenceLoop()
    {
        while (m_IsRunning)
        {
            try
            {
                Texture activeFrame = GetCurrentFrame();
                if (activeFrame != null && activeFrame.width > 32 && activeFrame.height > 32)
                {
                    if (m_ShowHandTracking && m_DetectorWorker != null && m_LandmarkerWorker != null)
                    {
                        await ProcessHandDetection(activeFrame);
                    }
                }

                await Awaitable.NextFrameAsync();
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception e)
            {
                Debug.LogException(e);
                await Awaitable.NextFrameAsync();
            }
        }
    }

    Texture GetCurrentFrame()
    {
        if (m_EditorWebCam != null && m_EditorWebCam.isPlaying)
            return m_EditorWebCam;

#if !UNITY_EDITOR
        if (arCameraManager != null && arCameraManager.TryAcquireLatestCpuImage(out XRCpuImage cpuImage))
        {
            using (cpuImage)
            {
                if (m_CameraFrameTexture == null || m_CameraFrameTexture.width != cpuImage.width || m_CameraFrameTexture.height != cpuImage.height)
                {
                    m_CameraFrameTexture = new Texture2D(cpuImage.width, cpuImage.height, TextureFormat.RGBA32, false);
                }

                var conversionParams = new XRCpuImage.ConversionParams
                {
                    inputRect = new RectInt(0, 0, cpuImage.width, cpuImage.height),
                    outputDimensions = new Vector2Int(cpuImage.width, cpuImage.height),
                    outputFormat = TextureFormat.RGBA32,
                    transformation = XRCpuImage.Transformation.None
                };

                int bufferSize = cpuImage.GetConvertedDataSize(conversionParams);
                var rawData = new NativeArray<byte>(bufferSize, Allocator.Temp);
                try
                {
                    cpuImage.Convert(conversionParams, rawData.Slice());
                    m_CameraFrameTexture.LoadRawTextureData(rawData);
                    m_CameraFrameTexture.Apply();
                    return m_CameraFrameTexture;
                }
                finally
                {
                    rawData.Dispose();
                }
            }
        }
#endif
        return null;
    }

    int GetCameraRotationAngle(Texture texture)
    {
        return GetTotalVideoAngle();
    }

    async Awaitable ProcessHandDetection(Texture texture)
    {
        int rotAngle = GetCameraRotationAngle(texture);
        float theta = rotAngle * Mathf.Deg2Rad;

        var size = Mathf.Max(texture.width, texture.height);
        var scale = size / (float)k_DetectorInputSize;

        float2x3 M;
        if (rotAngle != 0)
        {
            float2 texCenter = 0.5f * new float2(texture.width, texture.height);
            float2 tensorCenter = 0.5f * new float2(k_DetectorInputSize, k_DetectorInputSize);

            M = BlazeUtils.mul(
                BlazeUtils.TranslationMatrix(texCenter),
                BlazeUtils.mul(
                    BlazeUtils.RotationMatrix(theta),
                    BlazeUtils.mul(
                        BlazeUtils.ScaleMatrix(new float2(scale, -scale)),
                        BlazeUtils.TranslationMatrix(-tensorCenter)
                    )
                )
            );
        }
        else
        {
            M = BlazeUtils.mul(
                BlazeUtils.TranslationMatrix(0.5f * (new Vector2(texture.width, texture.height) + new Vector2(-size, size))),
                BlazeUtils.ScaleMatrix(new Vector2(scale, -scale))
            );
        }

        BlazeUtils.SampleImageAffine(texture, m_DetectorInput, M);
        m_DetectorWorker.Schedule(m_DetectorInput);

        var outputIdxAwaitable = (m_DetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
        var outputScoreAwaitable = (m_DetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
        var outputBoxAwaitable = (m_DetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

        using var outputIdx = await outputIdxAwaitable;
        using var outputScore = await outputScoreAwaitable;
        using var outputBox = await outputBoxAwaitable;

        int detectedCount = outputIdx.shape.length;
        string modeHint = $" | Mirror (M): {(mirrorHorizontally ? "ON" : "OFF")}";

        if (detectedCount == 0)
        {
            SetHandVisualsActive(false);
            if (statusText != null)
                statusText.text = $"Searching for hands... (Rot: {rotAngle}°){modeHint}";
            UpdateDistanceUI(0f, false, 0f, false, 0);
            return;
        }

        int handsToTrack = Mathf.Min(detectedCount, maxHands);

        // Sort detected hands by screen X coordinate for consistent Left/Right hand assignment
        var sortedDetections = new List<int>();
        for (int i = 0; i < detectedCount; i++) sortedDetections.Add(i);

        if (detectedCount >= 2)
        {
            sortedDetections.Sort((a, b) =>
            {
                int anchorA = outputIdx[a];
                int anchorB = outputIdx[b];
                float xA = k_DetectorInputSize * m_Anchors[anchorA, 0] + outputBox[0, a, 0];
                float xB = k_DetectorInputSize * m_Anchors[anchorB, 0] + outputBox[0, b, 0];
                return mirrorHorizontally ? xB.CompareTo(xA) : xA.CompareTo(xB);
            });
        }

        var canvas = FindAnyObjectByType<Canvas>();
        Vector2 bgSize = (cameraBackgroundUI != null)
            ? cameraBackgroundUI.rectTransform.sizeDelta
            : new Vector2(texture.width, texture.height);

        float[] handDepths = new float[k_MaxHands];
        bool[] handDepthFound = new bool[k_MaxHands];

        string topGesture = "Detecting...";
        for (int h = 0; h < handsToTrack; h++)
        {
            int detIdx = sortedDetections[h];
            int idx = outputIdx[detIdx];

            var anchorPos = k_DetectorInputSize * new float2(m_Anchors[idx, 0], m_Anchors[idx, 1]);
            var boxCenter = anchorPos + new float2(outputBox[0, detIdx, 0], outputBox[0, detIdx, 1]);
            var boxSize = math.max(outputBox[0, detIdx, 2], outputBox[0, detIdx, 3]);

            var kp0 = anchorPos + new float2(outputBox[0, detIdx, 4], outputBox[0, detIdx, 5]);
            var kp2 = anchorPos + new float2(outputBox[0, detIdx, 8], outputBox[0, detIdx, 9]);
            var delta = kp2 - kp0;
            var up = delta / math.max(0.0001f, math.length(delta));
            var thetaBox = math.atan2(delta.y, delta.x);
            var rot = 0.5f * Mathf.PI - thetaBox;
            boxCenter += 0.5f * boxSize * up;
            boxSize *= 2.6f;

            var origin2 = new float2(0.5f * k_LandmarkerInputSize, 0.5f * k_LandmarkerInputSize);
            var scale2 = boxSize / k_LandmarkerInputSize;
            var M2 = BlazeUtils.mul(M, BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.mul(
                BlazeUtils.TranslationMatrix(boxCenter),
                BlazeUtils.ScaleMatrix(new float2(scale2, -scale2))),
                BlazeUtils.RotationMatrix(rot)),
                BlazeUtils.TranslationMatrix(-origin2)));

            BlazeUtils.SampleImageAffine(texture, m_LandmarkerInput, M2);
            m_LandmarkerWorker.Schedule(m_LandmarkerInput);

            var landmarksAwaitable = (m_LandmarkerWorker.PeekOutput("Identity") as Tensor<float>).ReadbackAndCloneAsync();
            using var landmarks = await landmarksAwaitable;

            var keypointPositions = new Vector3[k_NumKeypoints];
            var uiPositions = new Vector2[k_NumKeypoints];
            var raw3D = new Vector3[k_NumKeypoints];
            Vector2 palmScreenPos = Vector2.zero;

            for (int i = 0; i < k_NumKeypoints; i++)
            {
                var pt = BlazeUtils.mul(M2, new float2(landmarks[3 * i], landmarks[3 * i + 1]));
                raw3D[i] = new Vector3(landmarks[3 * i], landmarks[3 * i + 1], landmarks[3 * i + 2]);

                // Normalized position in camera sensor space: [-0.5, 0.5]
                float normX = (pt.x - 0.5f * texture.width) / (float)texture.width;
                float normY = (pt.y - 0.5f * texture.height) / (float)texture.height;

                // 1:1 mapping through cameraBackgroundUI transform matrix
                // Guarantees exact lockstep alignment with camera video feed regardless of rotation/scale/mirror
                Vector3 localPoint = new Vector3(normX * bgSize.x, normY * bgSize.y, 0f);

                Vector3 worldPoint;
                Vector2 uiPos;
                if (cameraBackgroundUI != null)
                {
                    worldPoint = cameraBackgroundUI.transform.TransformPoint(localPoint);
                    uiPos = (m_UIHandRoots[h] != null)
                        ? (Vector2)m_UIHandRoots[h].InverseTransformPoint(worldPoint)
                        : (canvas != null ? (Vector2)canvas.transform.InverseTransformPoint(worldPoint) : (Vector2)worldPoint);
                }
                else
                {
                    uiPos = new Vector2(normX * Screen.width, normY * Screen.height);
                    worldPoint = new Vector3(uiPos.x, uiPos.y, 0f);
                }

                uiPositions[i] = uiPos;

                if (m_UIJoints[h] != null && m_UIJoints[h][i] != null)
                    m_UIJoints[h][i].anchoredPosition = uiPos;

                Vector2 screenPos = RectTransformUtility.WorldToScreenPoint(
                    (canvas != null && canvas.renderMode != RenderMode.ScreenSpaceOverlay) ? m_MainCamera : null,
                    worldPoint
                );

                if (i == 0 || i == 9)
                    palmScreenPos += screenPos * 0.5f;

                Vector3 worldPos;
                if (depthManager != null && depthManager.TryGetWorldPositionFromDepth(screenPos, out Vector3 arDepthPos))
                {
                    worldPos = arDepthPos;
                }
                else
                {
                    float relZ = landmarks[3 * i + 2];
                    float handDist = 0.65f - (relZ * 0.35f);
                    worldPos = m_MainCamera != null 
                        ? m_MainCamera.ScreenToWorldPoint(new Vector3(screenPos.x, screenPos.y, handDist))
                        : new Vector3(screenPos.x / Mathf.Max(1f, Screen.width), screenPos.y / Mathf.Max(1f, Screen.height), handDist);
                }

                keypointPositions[i] = worldPos;
                if (m_JointRenderers[h] != null && m_JointRenderers[h][i] != null && m_MainCamera != null)
                {
                    Vector3 camOffset = m_MainCamera.transform.forward * 0.002f;
                    m_JointRenderers[h][i].SetPosition(0, worldPos - camOffset);
                    m_JointRenderers[h][i].SetPosition(1, worldPos + camOffset);
                }
            }

            string gesture = ClassifyGesture(raw3D);
            if (h == 0) topGesture = gesture;

            // Update 3D VR Holographic Hand for slot h
            if (vrHandVisualizers != null && h < vrHandVisualizers.Length && vrHandVisualizers[h] != null && showVRHand)
            {
                vrHandVisualizers[h].UpdatePose(keypointPositions, m_MainCamera);
            }

            // Update UI Bone Lines for slot h
            for (int b = 0; b < HandBones.Length; b++)
            {
                var (startIdx, endIdx) = HandBones[b];
                if (m_UIBones[h] != null && m_UIBones[h][b] != null)
                {
                    Vector2 pA = uiPositions[startIdx];
                    Vector2 pB = uiPositions[endIdx];
                    Vector2 diff = pB - pA;
                    float dist = diff.magnitude;
                    float angle = Mathf.Atan2(diff.y, diff.x) * Mathf.Rad2Deg - 90f;

                    m_UIBones[h][b].anchoredPosition = pA;
                    m_UIBones[h][b].sizeDelta = new Vector2(4f, dist);
                    m_UIBones[h][b].localEulerAngles = new Vector3(0, 0, angle);
                }

                if (m_BoneRenderers[h] != null && m_BoneRenderers[h][b] != null)
                {
                    m_BoneRenderers[h][b].SetPosition(0, keypointPositions[startIdx]);
                    m_BoneRenderers[h][b].SetPosition(1, keypointPositions[endIdx]);
                }
            }

            // Calculate metric hand depth (meters)
            float metricDepth = 0.5f;
            if (depthManager != null && depthManager.TryGetDepthAtScreenPoint(palmScreenPos, out float arDepth) && arDepth > 0.05f)
            {
                metricDepth = arDepth;
                handDepthFound[h] = true;
            }
            else
            {
                // Geometric depth estimation from wrist-to-knuckle pixel span
                float palmPixelSpan = Vector2.Distance(uiPositions[0], uiPositions[9]);
                float refHeight = (canvas != null) ? canvas.GetComponent<RectTransform>().rect.height : Screen.height;
                metricDepth = (refHeight * 0.17f) / Mathf.Max(15f, palmPixelSpan);
                metricDepth = Mathf.Clamp(metricDepth, 0.20f, 2.50f);
                handDepthFound[h] = true;
            }
            handDepths[h] = metricDepth;

            // Depth tint on joints (closer = bright cyan/green, farther = magenta)
            float depthFactor = Mathf.Clamp01((metricDepth - 0.25f) / 0.85f);
            Color depthColor = Color.Lerp(new Color(0f, 1f, 0.85f, 1f), (h == 0) ? hand0Color : hand1Color, depthFactor);

            for (int i = 0; i < k_NumKeypoints; i++)
            {
                if (m_UIJoints[h] != null && m_UIJoints[h][i] != null)
                {
                    var img = m_UIJoints[h][i].GetComponent<Image>();
                    if (img != null) img.color = depthColor;
                }
            }

            // Update Floating Depth Badge
            if (m_UIDepthBadgeRects[h] != null && m_UIDepthBadges[h] != null)
            {
                m_UIDepthBadgeRects[h].anchoredPosition = uiPositions[0] + new Vector2(0f, -50f);
                m_UIDepthBadges[h].text = $"{gesture} | {metricDepth:F2}m";
                m_UIDepthBadgeRects[h].gameObject.SetActive(true);
            }

            SetHandVisualActive(h, true);
        }

        // Hide any visualizers for undetected hand slots
        for (int h = handsToTrack; h < k_MaxHands; h++)
        {
            SetHandVisualActive(h, false);
        }

        if (statusText != null)
        {
            statusText.text = handsToTrack >= 2
                ? $"Both Hands Tracked Active (2 Hands){modeHint}"
                : $"Hand Tracking Active (1 Hand){modeHint}";
        }

        UpdateDistanceUI(handDepths[0], handDepthFound[0], handDepths[1], handDepthFound[1], handsToTrack, topGesture);
    }

    void UpdateDistanceUI(float depth0, bool valid0, float depth1, bool valid1, int handsCount, string gesture = "")
    {
        if (distanceText == null) return;

        if (handsCount == 0)
        {
            distanceText.text = "Show your hand to track gestures";
        }
        else if (handsCount == 1)
        {
            string depthStr = (valid0 && depth0 > 0.05f) ? $" ({depth0:F2} m)" : "";
            distanceText.text = $"Gesture: {gesture}{depthStr}";
        }
        else
        {
            distanceText.text = $"Both Hands: {gesture}";
        }
    }

    string ClassifyGesture(Vector3[] lm)
    {
        if (lm == null || lm.Length < 21) return "Unknown";

        Vector3 wrist = lm[0];
        bool indexOpen = Vector3.Distance(lm[8], wrist) > Vector3.Distance(lm[6], wrist);
        bool middleOpen = Vector3.Distance(lm[12], wrist) > Vector3.Distance(lm[10], wrist);
        bool ringOpen = Vector3.Distance(lm[16], wrist) > Vector3.Distance(lm[14], wrist);
        bool pinkyOpen = Vector3.Distance(lm[20], wrist) > Vector3.Distance(lm[18], wrist);
        bool thumbOpen = Vector3.Distance(lm[4], lm[17]) > Vector3.Distance(lm[3], lm[17]);

        float pinchDist = Vector3.Distance(lm[4], lm[8]);
        float palmScale = Vector3.Distance(lm[0], lm[9]);
        bool isPinch = pinchDist < (palmScale * 0.35f);

        if (isPinch && middleOpen && ringOpen && pinkyOpen) return "👌 Pinch / OK";
        if (thumbOpen && indexOpen && middleOpen && ringOpen && pinkyOpen) return "🖐️ Open Palm";
        if (!thumbOpen && !indexOpen && !middleOpen && !ringOpen && !pinkyOpen) return "✊ Fist";
        if (!thumbOpen && indexOpen && middleOpen && !ringOpen && !pinkyOpen) return "✌️ Peace / V";
        if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen) return "👉 Pointing";
        if (thumbOpen && !indexOpen && !middleOpen && !ringOpen && !pinkyOpen) return "👍 Thumbs Up";
        if (thumbOpen && indexOpen && !middleOpen && !ringOpen && pinkyOpen) return "🤟 Rock On";

        return "✋ Hand Tracking";
    }

    void SetStatus(string msg)
    {
        if (statusText != null)
            statusText.text = msg;
    }

    void SetHandVisualActive(int h, bool active)
    {
        if (h < 0 || h >= k_MaxHands) return;

        bool showSkel = active && showSkeleton;
        bool showVR = active && showVRHand;

        if (m_UIHandRoots != null && h < m_UIHandRoots.Length && m_UIHandRoots[h] != null)
            m_UIHandRoots[h].gameObject.SetActive(showSkel);

        if (m_UIDepthBadgeRects != null && h < m_UIDepthBadgeRects.Length && m_UIDepthBadgeRects[h] != null)
            m_UIDepthBadgeRects[h].gameObject.SetActive(showSkel);

        if (m_HandVisualsRoots != null && h < m_HandVisualsRoots.Length && m_HandVisualsRoots[h] != null)
            m_HandVisualsRoots[h].SetActive(showSkel);

        if (vrHandVisualizers != null && h < vrHandVisualizers.Length && vrHandVisualizers[h] != null)
            vrHandVisualizers[h].SetHandVisible(showVR);
    }

    void SetHandVisualsActive(bool active)
    {
        for (int h = 0; h < k_MaxHands; h++)
        {
            SetHandVisualActive(h, active);
        }
    }

    void CreateLineMaterial()
    {
        var shader = Shader.Find("Universal Render Pipeline/Unlit")
                  ?? Shader.Find("Sprites/Default")
                  ?? Shader.Find("Unlit/Color");
        if (shader != null)
            m_LineMaterial = new Material(shader);
    }

    void CreateHandVisualizers()
    {
        for (int h = 0; h < k_MaxHands; h++)
        {
            Color jCol = (h == 0) ? jointColor : new Color(0.9f, 0.4f, 1f, 1f);
            Color bCol = (h == 0) ? boneColor : new Color(1f, 0.5f, 0.8f, 0.9f);

            var rootGo = new GameObject($"HandVisualizers_{h}");
            rootGo.transform.SetParent(transform, false);
            m_HandVisualsRoots[h] = rootGo;

            m_JointRenderers[h] = new LineRenderer[k_NumKeypoints];
            for (int i = 0; i < k_NumKeypoints; i++)
            {
                var go = new GameObject($"Joint_{i}");
                go.transform.SetParent(rootGo.transform, false);
                var lr = go.AddComponent<LineRenderer>();
                ConfigureLineRenderer(lr, jCol, 0.015f);
                lr.positionCount = 2;
                m_JointRenderers[h][i] = lr;
            }

            m_BoneRenderers[h] = new LineRenderer[HandBones.Length];
            for (int b = 0; b < HandBones.Length; b++)
            {
                var go = new GameObject($"Bone_{b}");
                go.transform.SetParent(rootGo.transform, false);
                var lr = go.AddComponent<LineRenderer>();
                ConfigureLineRenderer(lr, bCol, 0.007f);
                lr.positionCount = 2;
                m_BoneRenderers[h][b] = lr;
            }

            SetHandVisualActive(h, false);
        }
    }

    void CreateUIHandVisualizers()
    {
        var canvas = FindAnyObjectByType<Canvas>();
        if (canvas == null) return;

        for (int h = 0; h < k_MaxHands; h++)
        {
            Color jCol = (h == 0) ? jointColor : new Color(0.9f, 0.4f, 1f, 1f);
            Color bCol = (h == 0) ? boneColor : new Color(1f, 0.5f, 0.8f, 0.9f);

            var uiRootObj = new GameObject($"HandVisualsUI_{h}", typeof(RectTransform));
            uiRootObj.transform.SetParent(canvas.transform, false);
            uiRootObj.transform.SetSiblingIndex(1 + h);
            var rt = uiRootObj.GetComponent<RectTransform>();
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.pivot = new Vector2(0.5f, 0.5f);
            rt.anchoredPosition = Vector2.zero;
            rt.sizeDelta = Vector2.zero;
            rt.offsetMin = Vector2.zero;
            rt.offsetMax = Vector2.zero;
            m_UIHandRoots[h] = rt;

            m_UIJoints[h] = new RectTransform[k_NumKeypoints];
            for (int i = 0; i < k_NumKeypoints; i++)
            {
                var jointObj = new GameObject($"UIJoint_{i}", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
                jointObj.transform.SetParent(rt, false);
                var img = jointObj.GetComponent<Image>();
                img.color = jCol;
                var jrt = jointObj.GetComponent<RectTransform>();
                jrt.sizeDelta = new Vector2(16f, 16f);
                m_UIJoints[h][i] = jrt;
            }

            m_UIBones[h] = new RectTransform[HandBones.Length];
            for (int b = 0; b < HandBones.Length; b++)
            {
                var boneObj = new GameObject($"UIBone_{b}", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
                boneObj.transform.SetParent(rt, false);
                boneObj.transform.SetAsFirstSibling();
                var img = boneObj.GetComponent<Image>();
                img.color = bCol;
                var brt = boneObj.GetComponent<RectTransform>();
                brt.pivot = new Vector2(0.5f, 0f);
                m_UIBones[h][b] = brt;
            }

            // Floating Depth & Gesture Badge over Wrist
            var badgeObj = new GameObject($"DepthBadge_{h}", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image), typeof(Outline));
            badgeObj.transform.SetParent(rt, false);
            var bRt = badgeObj.GetComponent<RectTransform>();
            bRt.anchorMin = new Vector2(0.5f, 0.5f);
            bRt.anchorMax = new Vector2(0.5f, 0.5f);
            bRt.pivot = new Vector2(0.5f, 0.5f);
            bRt.sizeDelta = new Vector2(240f, 55f);
            m_UIDepthBadgeRects[h] = bRt;

            var bImg = badgeObj.GetComponent<Image>();
            bImg.color = new Color(0.06f, 0.08f, 0.14f, 0.90f);
            var bOutline = badgeObj.GetComponent<Outline>();
            bOutline.effectColor = (h == 0) ? hand0Color : hand1Color;
            bOutline.effectDistance = new Vector2(2f, -2f);

            var bTextObj = new GameObject("Text", typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
            bTextObj.transform.SetParent(badgeObj.transform, false);
            var btRt = bTextObj.GetComponent<RectTransform>();
            btRt.anchorMin = Vector2.zero;
            btRt.anchorMax = Vector2.one;
            btRt.sizeDelta = Vector2.zero;
            var bText = bTextObj.GetComponent<Text>();
            bText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
            bText.fontSize = 20;
            bText.fontStyle = FontStyle.Bold;
            bText.alignment = TextAnchor.MiddleCenter;
            bText.color = (h == 0) ? hand0Color : hand1Color;
            m_UIDepthBadges[h] = bText;
            badgeObj.SetActive(false);

            SetHandVisualActive(h, false);
        }
    }

    void ConfigureLineRenderer(LineRenderer lr, Color col, float width)
    {
        if (m_LineMaterial != null) lr.material = m_LineMaterial;
        lr.startColor = col;
        lr.endColor = col;
        lr.startWidth = width;
        lr.endWidth = width;
        lr.numCapVertices = 4;
        lr.useWorldSpace = true;
    }

    void Cleanup()
    {
        m_IsRunning = false;
        if (m_EditorWebCam != null && m_EditorWebCam.isPlaying)
            m_EditorWebCam.Stop();

        m_DetectorWorker?.Dispose();
        m_LandmarkerWorker?.Dispose();
        m_DetectorInput?.Dispose();
        m_LandmarkerInput?.Dispose();

        if (m_CameraFrameTexture != null)
            Destroy(m_CameraFrameTexture);
    }

    void OnDestroy()
    {
        Cleanup();
    }
}
