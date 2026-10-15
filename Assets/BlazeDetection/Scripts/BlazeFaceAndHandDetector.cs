using System;
using System.Collections.Generic;
using Unity.Mathematics;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// Unified Blaze Face & Hand detector for Unity Sentis (com.unity.ai.inference).
/// Supports both Webcam live video and static/dynamic Textures.
/// Auto-generates visualization lines/points at runtime if prefabs are not manually assigned.
/// </summary>
public class BlazeFaceAndHandDetector : MonoBehaviour
{
    [Header("Input Source")]
    [Tooltip("Enable to capture from the default WebCam")]
    public bool useWebcam = true;
    public int webcamWidth = 1280;
    public int webcamHeight = 720;
    public int webcamFPS = 30;
    [Tooltip("Static texture to use if webcam is disabled or unavailable")]
    public Texture staticTexture;
    public ImagePreview imagePreview;

    [Header("Detection Controls")]
    public bool detectFace = true;
    public bool detectHand = true;

    [Header("BlazeFace Configuration")]
    public ModelAsset faceDetectorModelAsset;
    public TextAsset faceAnchorsCSV;
    [Range(0.01f, 1f)] public float faceIouThreshold = 0.3f;
    [Range(0.01f, 1f)] public float faceScoreThreshold = 0.5f;
    public int maxFaceCount = 4;
    public FacePreview[] facePreviews;

    [Header("BlazeHand Configuration")]
    public ModelAsset handDetectorModelAsset;
    public ModelAsset handLandmarkerModelAsset;
    public TextAsset handAnchorsCSV;
    [Range(0.01f, 1f)] public float handScoreThreshold = 0.5f;
    public HandPreview handPreview;

    [Header("Visual Styling")]
    public Color faceBoxColor = new Color(0.2f, 1f, 0.2f, 0.9f);
    public Color faceKeypointColor = new Color(1f, 0.3f, 0.3f, 0.9f);
    public Color handJointColor = new Color(0.2f, 0.8f, 1f, 0.9f);
    public Color handBoneColor = new Color(1f, 0.85f, 0.2f, 0.8f);

    // Model Constants
    const int k_NumFaceAnchors = 896;
    const int k_FaceInputSize = 128;
    const int k_NumFaceKeypoints = 6;

    const int k_NumHandAnchors = 2016;
    const int k_HandDetectorInputSize = 192;
    const int k_HandLandmarkerInputSize = 224;
    const int k_NumHandKeypoints = 21;

    // Hand connections (MediaPipe Hand 21 landmarks)
    static readonly (int, int)[] HandConnections = new (int, int)[]
    {
        // Palm
        (0, 1), (1, 2), (2, 3), (3, 4),        // Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        // Index
        (5, 9), (9, 10), (10, 11), (11, 12),   // Middle
        (9, 13), (13, 14), (14, 15), (15, 16), // Ring
        (13, 17), (17, 18), (18, 19), (19, 20),// Pinky
        (0, 17)                                // Palm base
    };

    // Runtime state
    WebCamTexture m_WebCamTexture;
    float[,] m_FaceAnchors;
    float[,] m_HandAnchors;

    Worker m_FaceDetectorWorker;
    Worker m_HandDetectorWorker;
    Worker m_HandLandmarkerWorker;

    Tensor<float> m_FaceInputTensor;
    Tensor<float> m_HandDetectorInputTensor;
    Tensor<float> m_HandLandmarkerInputTensor;

    Material m_LineMaterial;
    List<LineRenderer> m_HandBoneRenderers = new List<LineRenderer>();

    bool m_IsRunning;
    float m_TextureWidth = 1280f;
    float m_TextureHeight = 720f;

    public WebCamTexture ActiveWebcam => m_WebCamTexture;

    void Awake()
    {
        var cam = Camera.main;
        if (cam != null)
        {
            cam.transform.position = new Vector3(0, 0, 0);
            cam.transform.rotation = Quaternion.identity;
            cam.orthographic = true;
            cam.orthographicSize = 0.5f;
            cam.nearClipPlane = 0.01f;
            cam.farClipPlane = 50f;
        }

        CreateDefaultLineMaterial();
        SetupDefaultVisualizers();
    }

    async void Start()
    {
        // Auto-load models & anchors from Resources/Data if unassigned
        AutoAssignAssetsIfNull();

#if UNITY_ANDROID
        if (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Camera))
        {
            UnityEngine.Android.Permission.RequestUserPermission(UnityEngine.Android.Permission.Camera);
            int timeout = 50;
            while (!UnityEngine.Android.Permission.HasUserAuthorizedPermission(UnityEngine.Android.Permission.Camera) && timeout > 0)
            {
                await Awaitable.WaitForSecondsAsync(0.1f);
                timeout--;
            }
        }
#endif

        // 1. Setup Input Feed
        if (useWebcam)
        {
            var devices = WebCamTexture.devices;
            if (devices != null && devices.Length > 0)
            {
                m_WebCamTexture = new WebCamTexture(devices[0].name, webcamWidth, webcamHeight, webcamFPS);
                m_WebCamTexture.Play();
            }
            else
            {
                Debug.LogWarning("[Blaze] No webcam device found. Falling back to static texture.");
                useWebcam = false;
            }
        }

        // 2. Initialize Neural Network Models
        InitModels();

        // 3. Inference Loop
        m_IsRunning = true;
        while (m_IsRunning)
        {
            try
            {
                var currentTexture = GetActiveTexture();
                if (currentTexture != null && currentTexture.width > 16 && currentTexture.height > 16)
                {
                    m_TextureWidth = currentTexture.width;
                    m_TextureHeight = currentTexture.height;

                    if (imagePreview != null)
                        imagePreview.SetTexture(currentTexture);

                    if (detectFace && m_FaceDetectorWorker != null)
                    {
                        await RunFaceDetection(currentTexture);
                    }

                    if (detectHand && m_HandDetectorWorker != null && m_HandLandmarkerWorker != null)
                    {
                        await RunHandDetection(currentTexture);
                    }
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

        Cleanup();
    }

    void AutoAssignAssetsIfNull()
    {
        if (faceAnchorsCSV == null)
            faceAnchorsCSV = Resources.Load<TextAsset>("Data/face_anchors");
        if (handAnchorsCSV == null)
            handAnchorsCSV = Resources.Load<TextAsset>("Data/hand_anchors");
    }

    Texture GetActiveTexture()
    {
        if (useWebcam && m_WebCamTexture != null && m_WebCamTexture.isPlaying)
            return m_WebCamTexture;
        return staticTexture;
    }

    void CreateDefaultLineMaterial()
    {
        var shader = Shader.Find("Universal Render Pipeline/Unlit")
                  ?? Shader.Find("Sprites/Default")
                  ?? Shader.Find("Unlit/Color");
        if (shader != null)
            m_LineMaterial = new Material(shader);
    }

    void SetupDefaultVisualizers()
    {
        // 1. Setup Background Image Quad if needed
        if (imagePreview == null)
        {
            var previewGO = new GameObject("BlazeImagePreview");
            previewGO.transform.SetParent(transform, false);
            var quad = GameObject.CreatePrimitive(PrimitiveType.Quad);
            quad.name = "BackgroundQuad";
            quad.transform.SetParent(previewGO.transform, false);
            quad.transform.localPosition = new Vector3(0, 0, 1f); // slightly behind detection markers
            var collider = quad.GetComponent<Collider>();
            if (collider != null) Destroy(collider);

            var renderer = quad.GetComponent<MeshRenderer>();
            if (renderer != null && m_LineMaterial != null)
                renderer.material = new Material(m_LineMaterial);

            imagePreview = previewGO.AddComponent<ImagePreview>();
            imagePreview.imageQuad = quad;
        }

        // 2. Setup Face Previews if unassigned
        if (facePreviews == null || facePreviews.Length == 0)
        {
            var faceList = new List<FacePreview>();
            var facesRoot = new GameObject("BlazeFacePreviews");
            facesRoot.transform.SetParent(transform, false);

            for (int i = 0; i < maxFaceCount; i++)
            {
                var faceGO = new GameObject($"FacePreview_{i}");
                faceGO.transform.SetParent(facesRoot.transform, false);
                var fp = faceGO.AddComponent<FacePreview>();

                // BoundingBox
                var boxGO = new GameObject("BoundingBox");
                boxGO.transform.SetParent(faceGO.transform, false);
                var lr = boxGO.AddComponent<LineRenderer>();
                ConfigureLineRenderer(lr, faceBoxColor, 0.008f);
                var bb = boxGO.AddComponent<BoundingBox>();
                bb.lineRenderer = lr;
                bb.color = faceBoxColor;
                fp.boundingBox = bb;

                // 6 Keypoints
                fp.keypoints = new Keypoint[k_NumFaceKeypoints];
                for (int k = 0; k < k_NumFaceKeypoints; k++)
                {
                    var kpGO = new GameObject($"FaceKeypoint_{k}");
                    kpGO.transform.SetParent(faceGO.transform, false);
                    var kplr = kpGO.AddComponent<LineRenderer>();
                    ConfigureLineRenderer(kplr, faceKeypointColor, 0.015f);
                    var kp = kpGO.AddComponent<Keypoint>();
                    kp.innerCircle = kplr;
                    kp.innerColor = faceKeypointColor;
                    fp.keypoints[k] = kp;
                }

                faceGO.SetActive(false);
                faceList.Add(fp);
            }

            facePreviews = faceList.ToArray();
        }

        // 3. Setup Hand Preview if unassigned
        if (handPreview == null)
        {
            var handGO = new GameObject("BlazeHandPreview");
            handGO.transform.SetParent(transform, false);
            handPreview = handGO.AddComponent<HandPreview>();

            handPreview.keypoints = new Keypoint[k_NumHandKeypoints];
            for (int k = 0; k < k_NumHandKeypoints; k++)
            {
                var kpGO = new GameObject($"HandKeypoint_{k}");
                kpGO.transform.SetParent(handGO.transform, false);
                var lr = kpGO.AddComponent<LineRenderer>();
                ConfigureLineRenderer(lr, handJointColor, 0.018f);
                var kp = kpGO.AddComponent<Keypoint>();
                kp.innerCircle = lr;
                kp.innerColor = handJointColor;
                handPreview.keypoints[k] = kp;
            }

            // Create bone connecting lines
            var bonesRoot = new GameObject("HandBones");
            bonesRoot.transform.SetParent(handGO.transform, false);
            for (int b = 0; b < HandConnections.Length; b++)
            {
                var boneGO = new GameObject($"Bone_{b}");
                boneGO.transform.SetParent(bonesRoot.transform, false);
                var lr = boneGO.AddComponent<LineRenderer>();
                ConfigureLineRenderer(lr, handBoneColor, 0.006f);
                lr.positionCount = 2;
                m_HandBoneRenderers.Add(lr);
            }

            handGO.SetActive(false);
        }
    }

    void ConfigureLineRenderer(LineRenderer lr, Color color, float width)
    {
        if (m_LineMaterial != null)
            lr.material = m_LineMaterial;
        lr.startColor = color;
        lr.endColor = color;
        lr.startWidth = width;
        lr.endWidth = width;
        lr.useWorldSpace = true;
    }

    void InitModels()
    {
        // 1. Initialize Face Model
        if (detectFace && faceDetectorModelAsset != null && faceAnchorsCSV != null)
        {
            m_FaceAnchors = BlazeUtils.LoadAnchors(faceAnchorsCSV.text, k_NumFaceAnchors);
            var faceModel = ModelLoader.Load(faceDetectorModelAsset);

            var graph = new FunctionalGraph();
            var input = graph.AddInput(faceModel, 0);
            var outputs = Functional.Forward(faceModel, 2 * input - 1);
            var boxes = outputs[0];
            var scores = outputs[1];
            var anchorsData = new float[k_NumFaceAnchors * 4];
            Buffer.BlockCopy(m_FaceAnchors, 0, anchorsData, 0, anchorsData.Length * sizeof(float));
            var anchors = Functional.Constant(new TensorShape(k_NumFaceAnchors, 4), anchorsData);
            var filtered = BlazeUtils.NMSFiltering(boxes, scores, anchors, k_FaceInputSize, faceIouThreshold, faceScoreThreshold);
            faceModel = graph.Compile(filtered.Item1, filtered.Item2, filtered.Item3);

            m_FaceDetectorWorker = new Worker(faceModel, BackendType.GPUCompute);
            m_FaceInputTensor = new Tensor<float>(new TensorShape(1, k_FaceInputSize, k_FaceInputSize, 3));
        }

        // 2. Initialize Hand Models
        if (detectHand && handDetectorModelAsset != null && handLandmarkerModelAsset != null && handAnchorsCSV != null)
        {
            m_HandAnchors = BlazeUtils.LoadAnchors(handAnchorsCSV.text, k_NumHandAnchors);

            var detectorModel = ModelLoader.Load(handDetectorModelAsset);
            var graph = new FunctionalGraph();
            var input = graph.AddInput(detectorModel, 0);
            var outputs = Functional.Forward(detectorModel, input);
            var filtered = BlazeUtils.ArgMaxFiltering(outputs[0], outputs[1]);
            detectorModel = graph.Compile(filtered.Item1, filtered.Item2, filtered.Item3);

            m_HandDetectorWorker = new Worker(detectorModel, BackendType.GPUCompute);

            var landmarkerModel = ModelLoader.Load(handLandmarkerModelAsset);
            m_HandLandmarkerWorker = new Worker(landmarkerModel, BackendType.GPUCompute);

            m_HandDetectorInputTensor = new Tensor<float>(new TensorShape(1, k_HandDetectorInputSize, k_HandDetectorInputSize, 3));
            m_HandLandmarkerInputTensor = new Tensor<float>(new TensorShape(1, k_HandLandmarkerInputSize, k_HandLandmarkerInputSize, 3));
        }
    }

    Vector3 ImageToWorld(Vector2 position)
    {
        return (position - 0.5f * new Vector2(m_TextureWidth, m_TextureHeight)) / m_TextureHeight;
    }

    async Awaitable RunFaceDetection(Texture texture)
    {
        var size = Mathf.Max(texture.width, texture.height);
        var scale = size / (float)k_FaceInputSize;
        var M = BlazeUtils.mul(
            BlazeUtils.TranslationMatrix(0.5f * (new Vector2(texture.width, texture.height) + new Vector2(-size, size))),
            BlazeUtils.ScaleMatrix(new Vector2(scale, -scale))
        );

        BlazeUtils.SampleImageAffine(texture, m_FaceInputTensor, M);
        m_FaceDetectorWorker.Schedule(m_FaceInputTensor);

        var outputIndicesAwaitable = (m_FaceDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
        var outputScoresAwaitable = (m_FaceDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
        var outputBoxesAwaitable = (m_FaceDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

        using var outputIndices = await outputIndicesAwaitable;
        using var outputScores = await outputScoresAwaitable;
        using var outputBoxes = await outputBoxesAwaitable;

        var numFaces = outputIndices.shape.length;
        if (facePreviews != null)
        {
            for (var i = 0; i < facePreviews.Length; i++)
            {
                if (facePreviews[i] == null) continue;
                var active = i < numFaces;
                facePreviews[i].SetActive(active);
                if (!active) continue;

                var idx = outputIndices[i];
                var anchorPosition = k_FaceInputSize * new float2(m_FaceAnchors[idx, 0], m_FaceAnchors[idx, 1]);
                var box_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBoxes[0, i, 0], outputBoxes[0, i, 1]));
                var boxTopRight_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBoxes[0, i, 0] + 0.5f * outputBoxes[0, i, 2], outputBoxes[0, i, 1] + 0.5f * outputBoxes[0, i, 3]));
                var boxSize = 2f * (boxTopRight_ImageSpace - box_ImageSpace);

                facePreviews[i].SetBoundingBox(true, ImageToWorld(box_ImageSpace), boxSize / texture.height);

                for (var j = 0; j < k_NumFaceKeypoints; j++)
                {
                    var kp_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBoxes[0, i, 4 + 2 * j + 0], outputBoxes[0, i, 4 + 2 * j + 1]));
                    facePreviews[i].SetKeypoint(j, true, ImageToWorld(kp_ImageSpace));
                }
            }
        }
    }

    async Awaitable RunHandDetection(Texture texture)
    {
        var webCam = texture as WebCamTexture;
        int rotAngle = webCam != null ? webCam.videoRotationAngle : 0;
        float thetaRot = rotAngle * Mathf.Deg2Rad;

        var size = Mathf.Max(texture.width, texture.height);
        var scale = size / (float)k_HandDetectorInputSize;
        float2 center = 0.5f * new float2(texture.width, texture.height);
        float2 origin = 0.5f * new float2(k_HandDetectorInputSize, k_HandDetectorInputSize);

        var M = BlazeUtils.mul(
            BlazeUtils.TranslationMatrix(center),
            BlazeUtils.mul(
                BlazeUtils.RotationMatrix(thetaRot),
                BlazeUtils.mul(
                    BlazeUtils.ScaleMatrix(new float2(scale, -scale)),
                    BlazeUtils.TranslationMatrix(-origin)
                )
            )
        );

        BlazeUtils.SampleImageAffine(texture, m_HandDetectorInputTensor, M);
        m_HandDetectorWorker.Schedule(m_HandDetectorInputTensor);

        var outputIdxAwaitable = (m_HandDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
        var outputScoreAwaitable = (m_HandDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
        var outputBoxAwaitable = (m_HandDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

        using var outputIdx = await outputIdxAwaitable;
        using var outputScore = await outputScoreAwaitable;
        using var outputBox = await outputBoxAwaitable;

        var scorePasses = outputScore[0] >= handScoreThreshold;
        if (handPreview != null)
            handPreview.SetActive(scorePasses);

        for (int b = 0; b < m_HandBoneRenderers.Count; b++)
        {
            if (m_HandBoneRenderers[b] != null)
                m_HandBoneRenderers[b].enabled = scorePasses;
        }

        if (!scorePasses) return;

        var idx = outputIdx[0];
        var anchorPosition = k_HandDetectorInputSize * new float2(m_HandAnchors[idx, 0], m_HandAnchors[idx, 1]);
        var boxCentre_TensorSpace = anchorPosition + new float2(outputBox[0, 0, 0], outputBox[0, 0, 1]);
        var boxSize_TensorSpace = math.max(outputBox[0, 0, 2], outputBox[0, 0, 3]);

        var kp0_TensorSpace = anchorPosition + new float2(outputBox[0, 0, 4 + 2 * 0 + 0], outputBox[0, 0, 4 + 2 * 0 + 1]);
        var kp2_TensorSpace = anchorPosition + new float2(outputBox[0, 0, 4 + 2 * 2 + 0], outputBox[0, 0, 4 + 2 * 2 + 1]);
        var delta_TensorSpace = kp2_TensorSpace - kp0_TensorSpace;
        var up_TensorSpace = delta_TensorSpace / math.length(delta_TensorSpace);
        var theta = math.atan2(delta_TensorSpace.y, delta_TensorSpace.x);
        var rotation = 0.5f * Mathf.PI - theta;
        boxCentre_TensorSpace += 0.5f * boxSize_TensorSpace * up_TensorSpace;
        boxSize_TensorSpace *= 2.6f;

        var origin2 = new float2(0.5f * k_HandLandmarkerInputSize, 0.5f * k_HandLandmarkerInputSize);
        var scale2 = boxSize_TensorSpace / k_HandLandmarkerInputSize;
        var M2 = BlazeUtils.mul(M, BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.mul(
            BlazeUtils.TranslationMatrix(boxCentre_TensorSpace),
            BlazeUtils.ScaleMatrix(new float2(scale2, -scale2))),
            BlazeUtils.RotationMatrix(rotation)),
            BlazeUtils.TranslationMatrix(-origin2)));

        BlazeUtils.SampleImageAffine(texture, m_HandLandmarkerInputTensor, M2);
        m_HandLandmarkerWorker.Schedule(m_HandLandmarkerInputTensor);

        var landmarksAwaitable = (m_HandLandmarkerWorker.PeekOutput("Identity") as Tensor<float>).ReadbackAndCloneAsync();
        using var landmarks = await landmarksAwaitable;

        var landmarkPositions = new Vector3[k_NumHandKeypoints];

        if (handPreview != null)
        {
            for (var i = 0; i < k_NumHandKeypoints; i++)
            {
                var position_ImageSpace = BlazeUtils.mul(M2, new float2(landmarks[3 * i + 0], landmarks[3 * i + 1]));
                landmarkPositions[i] = ImageToWorld(position_ImageSpace) + new Vector3(0, 0, landmarks[3 * i + 2] / m_TextureHeight);
                handPreview.SetKeypoint(i, true, landmarkPositions[i]);
            }
        }

        // Update bone line renderers
        for (int b = 0; b < HandConnections.Length && b < m_HandBoneRenderers.Count; b++)
        {
            var (startIdx, endIdx) = HandConnections[b];
            var lr = m_HandBoneRenderers[b];
            if (lr != null)
            {
                lr.SetPosition(0, landmarkPositions[startIdx]);
                lr.SetPosition(1, landmarkPositions[endIdx]);
            }
        }
    }

    void Cleanup()
    {
        if (m_WebCamTexture != null && m_WebCamTexture.isPlaying)
        {
            m_WebCamTexture.Stop();
        }

        m_FaceDetectorWorker?.Dispose();
        m_FaceDetectorWorker = null;
        m_HandDetectorWorker?.Dispose();
        m_HandDetectorWorker = null;
        m_HandLandmarkerWorker?.Dispose();
        m_HandLandmarkerWorker = null;

        m_FaceInputTensor?.Dispose();
        m_FaceInputTensor = null;
        m_HandDetectorInputTensor?.Dispose();
        m_HandDetectorInputTensor = null;
        m_HandLandmarkerInputTensor?.Dispose();
        m_HandLandmarkerInputTensor = null;
    }

    void OnDestroy()
    {
        m_IsRunning = false;
        Cleanup();
    }
}
