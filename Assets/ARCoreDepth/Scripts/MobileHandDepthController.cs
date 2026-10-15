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

    [Tooltip("Camera sensor rotation angle on mobile devices (default 90 for standard Android back camera in portrait)")]
    public int cameraRotationAngle = 90;

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
    public bool showSkeleton = false;
    public Color hand0Color = new Color(0.1f, 0.85f, 1f, 0.85f);  // Cyan
    public Color hand1Color = new Color(0.85f, 0.3f, 1f, 0.85f);  // Neon Magenta / Violet

    [Header("Visual Styling (Skeleton Mode)")]
    public Color jointColor = new Color(0.2f, 0.85f, 1f, 1f);
    public Color boneColor = new Color(1f, 0.9f, 0.2f, 0.9f);

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

    void Start()
    {
        if (permissionHelper != null)
        {
            permissionHelper.OnPermissionGranted += OnPermissionGranted;
            permissionHelper.OnPermissionDenied += OnPermissionDenied;
        }
        else
        {
            OnPermissionGranted();
        }
    }

    void Update()
    {
#if UNITY_EDITOR
        UpdateWebCamOrientation();
#endif

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
            toggleDepthButton.onClick.AddListener(() =>
            {
                m_ShowDepthMap = !m_ShowDepthMap;
                if (depthPreviewUI != null)
                    depthPreviewUI.gameObject.SetActive(m_ShowDepthMap);
            });
        }

        if (toggleHandButton != null)
        {
            toggleHandButton.onClick.AddListener(() =>
            {
                // Cycle visual styles: VR Hologram Hand -> Both -> 2D Skeleton
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
#if UNITY_EDITOR
        if (WebCamTexture.devices.Length > 0)
        {
            m_EditorWebCam = new WebCamTexture(1280, 720, 30);
            m_EditorWebCam.Play();
            SetupEditorCameraBackground();
            SetStatus($"Running in Editor (Webcam: {WebCamTexture.devices[0].name})");
        }
        else
        {
            SetStatus("No webcam detected in Editor");
        }
#else
        SetStatus("ARCore active. Detecting hands & depth...");
#endif

        m_IsRunning = true;
        RunInferenceLoop();
    }

    void SetupEditorCameraBackground()
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
                var newBg = new GameObject("CameraBackgroundUI", typeof(RectTransform), typeof(CanvasRenderer), typeof(RawImage), typeof(AspectRatioFitter));
                newBg.transform.SetParent(canvas.transform, false);
                newBg.transform.SetAsFirstSibling();
                cameraBackgroundUI = newBg.GetComponent<RawImage>();
            }
        }

        if (cameraBackgroundUI != null && m_EditorWebCam != null)
        {
            cameraBackgroundUI.texture = m_EditorWebCam;
            cameraBackgroundUI.color = Color.white;
            cameraBackgroundUI.gameObject.SetActive(true);
            UpdateWebCamOrientation();
        }
    }

    void UpdateWebCamOrientation()
    {
        if (m_EditorWebCam == null || cameraBackgroundUI == null || m_EditorWebCam.width < 100) return;

        var rect = cameraBackgroundUI.rectTransform;
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.sizeDelta = Vector2.zero;
        rect.anchoredPosition = Vector2.zero;
        rect.localEulerAngles = Vector3.zero;

        var fitter = cameraBackgroundUI.GetComponent<AspectRatioFitter>();
        if (fitter == null) fitter = cameraBackgroundUI.gameObject.AddComponent<AspectRatioFitter>();
        fitter.aspectMode = AspectRatioFitter.AspectMode.EnvelopeParent;

        fitter.aspectRatio = (float)m_EditorWebCam.width / Mathf.Max(1, m_EditorWebCam.height);

        // Apply mirroring and vertical flip directly via localScale
        cameraBackgroundUI.transform.localScale = new Vector3(
            mirrorHorizontally ? -1f : 1f,
            flipVertically ? -1f : 1f,
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
#if UNITY_EDITOR
        if (m_EditorWebCam != null && m_EditorWebCam.isPlaying)
            return m_EditorWebCam;
#else
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
#if UNITY_EDITOR
        if (m_EditorWebCam != null)
            return m_EditorWebCam.videoRotationAngle;
        return 0;
#elif UNITY_ANDROID || UNITY_IOS
        if (Screen.height > Screen.width && texture.width > texture.height)
        {
            return cameraRotationAngle;
        }
        return 0;
#else
        return 0;
#endif
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

        // Viewport scale mapping for camera background
        float screenW = Screen.width;
        float screenH = Screen.height;
        float screenAspect = screenW / Mathf.Max(1f, screenH);

        float camAspect;
        if (rotAngle == 90 || rotAngle == 270)
        {
            camAspect = (float)texture.height / Mathf.Max(1f, texture.width);
        }
        else
        {
            camAspect = (float)texture.width / Mathf.Max(1f, texture.height);
        }

        float renderW, renderH;
        if (screenAspect > camAspect)
        {
            renderW = screenW;
            renderH = screenW / camAspect;
        }
        else
        {
            renderH = screenH;
            renderW = screenH * camAspect;
        }

        var canvas = FindAnyObjectByType<Canvas>();
        float toCanvas = 1f;
        if (canvas != null)
        {
            var canvasRt = canvas.GetComponent<RectTransform>();
            if (canvasRt != null && screenW > 0)
                toCanvas = canvasRt.rect.width / screenW;
        }

        float[] handDepths = new float[k_MaxHands];
        bool[] handDepthFound = new bool[k_MaxHands];

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
            Vector2 palmScreenPos = Vector2.zero;

            for (int i = 0; i < k_NumKeypoints; i++)
            {
                var pt = BlazeUtils.mul(M2, new float2(landmarks[3 * i], landmarks[3 * i + 1]));

                float uRaw = (pt.x / (float)texture.width) - 0.5f;
                float vRaw = (pt.y / (float)texture.height) - 0.5f;

                float u, v;
                if (rotAngle != 0)
                {
                    float cosT = Mathf.Cos(-theta);
                    float sinT = Mathf.Sin(-theta);
                    u = uRaw * cosT - vRaw * sinT;
                    v = uRaw * sinT + vRaw * cosT;
                }
                else
                {
                    u = uRaw;
                    v = vRaw;
                }

                float uiX = (mirrorHorizontally ? -u : u) * renderW;
                float uiY = (flipVertically ? -v : v) * renderH;

                Vector2 uiPos = new Vector2(uiX * toCanvas, uiY * toCanvas);
                uiPositions[i] = uiPos;

                if (m_UIJoints[h] != null && m_UIJoints[h][i] != null)
                    m_UIJoints[h][i].anchoredPosition = uiPos;

                float screenX = uiX + screenW * 0.5f;
                float screenY = uiY + screenH * 0.5f;

                if (i == 0 || i == 9)
                    palmScreenPos += new Vector2(screenX, screenY) * 0.5f;

                Vector3 worldPos;
                if (depthManager != null && depthManager.TryGetWorldPositionFromDepth(new Vector2(screenX, screenY), out Vector3 arDepthPos))
                {
                    worldPos = arDepthPos;
                }
                else
                {
                    float relZ = landmarks[3 * i + 2];
                    float handDist = 0.65f - (relZ * 0.35f);
                    worldPos = m_MainCamera != null 
                        ? m_MainCamera.ScreenToWorldPoint(new Vector3(screenX, screenY, handDist))
                        : new Vector3(screenX / screenW, screenY / screenH, handDist);
                }

                keypointPositions[i] = worldPos;
                if (m_JointRenderers[h] != null && m_JointRenderers[h][i] != null && m_MainCamera != null)
                {
                    Vector3 camOffset = m_MainCamera.transform.forward * 0.002f;
                    m_JointRenderers[h][i].SetPosition(0, worldPos - camOffset);
                    m_JointRenderers[h][i].SetPosition(1, worldPos + camOffset);
                }
            }

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

            SetHandVisualActive(h, true);

            if (depthManager != null)
            {
                handDepthFound[h] = depthManager.TryGetDepthAtScreenPoint(palmScreenPos, out handDepths[h]);
            }
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

        UpdateDistanceUI(handDepths[0], handDepthFound[0], handDepths[1], handDepthFound[1], handsToTrack);
    }

    void UpdateDistanceUI(float depth0, bool valid0, float depth1, bool valid1, int handsCount)
    {
        if (distanceText == null) return;

        if (handsCount == 0)
        {
            distanceText.text = "Hand Depth: Detecting...";
        }
        else if (handsCount == 1)
        {
            if (valid0 && depth0 > 0.05f)
                distanceText.text = $"Hand Depth: {depth0:F2} m";
            else
                distanceText.text = "Hand Depth: Estimating...";
        }
        else
        {
            string s0 = (valid0 && depth0 > 0.05f) ? $"{depth0:F2}m" : "OK";
            string s1 = (valid1 && depth1 > 0.05f) ? $"{depth1:F2}m" : "OK";
            distanceText.text = $"Left Hand: {s0} | Right Hand: {s1}";
        }
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
