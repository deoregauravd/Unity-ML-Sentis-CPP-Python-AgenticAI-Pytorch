using System;
using Unity.Collections;
using UnityEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;

/// <summary>
/// Manages ARCore Depth Mapping using Unity AR Foundation and the ARCore Depth API.
/// Enables environment depth sensing, physical distance queries in meters, 
/// and depth map visualization.
/// </summary>
[RequireComponent(typeof(AROcclusionManager))]
public class ARCoreDepthManager : MonoBehaviour
{
    [Header("Occlusion & Depth Settings")]
    [Tooltip("Requested Environment Depth mode (e.g., Best for high quality, Fastest for lower latency)")]
    public EnvironmentDepthMode requestedDepthMode = EnvironmentDepthMode.Best;

    [Tooltip("Preference for virtual objects occlusion against real-world geometry")]
    public OcclusionPreferenceMode occlusionPreference = OcclusionPreferenceMode.PreferEnvironmentOcclusion;

    [Header("Visualization (Optional)")]
    [Tooltip("UI RawImage or MeshRenderer to display the depth map preview")]
    public UnityEngine.UI.RawImage depthMapDisplay;

    [Tooltip("Material used to visualize depth (e.g., Unlit depth shader)")]
    public Material depthVisualizationMaterial;

    // Component References
    AROcclusionManager m_OcclusionManager;
    Camera m_ARCamera;

    // Callbacks & Events
    public event Action<Texture2D> OnDepthUpdated;

    public bool IsDepthSupported => m_OcclusionManager != null 
        && m_OcclusionManager.subsystem != null 
        && m_OcclusionManager.subsystem.running;

    public EnvironmentDepthMode CurrentDepthMode => m_OcclusionManager != null 
        ? m_OcclusionManager.currentEnvironmentDepthMode 
        : EnvironmentDepthMode.Disabled;

    public Texture2D EnvironmentDepthTexture => m_OcclusionManager != null 
        ? m_OcclusionManager.environmentDepthTexture 
        : null;

    void Awake()
    {
        m_OcclusionManager = GetComponent<AROcclusionManager>();
        m_ARCamera = GetComponent<Camera>();
        if (m_ARCamera == null)
            m_ARCamera = Camera.main;
    }

    void OnEnable()
    {
        if (m_OcclusionManager != null)
        {
            m_OcclusionManager.requestedEnvironmentDepthMode = requestedDepthMode;
            m_OcclusionManager.requestedOcclusionPreferenceMode = occlusionPreference;
        }
    }

    void Update()
    {
        if (m_OcclusionManager == null) return;

        // Update visualization if assigned
        var depthTexture = m_OcclusionManager.environmentDepthTexture;
        if (depthTexture != null)
        {
            if (depthMapDisplay != null)
            {
                depthMapDisplay.texture = depthTexture;
                if (depthVisualizationMaterial == null)
                {
                    var shader = Shader.Find("ARCore/DepthVisualizer");
                    if (shader != null)
                        depthVisualizationMaterial = new Material(shader);
                }
                if (depthVisualizationMaterial != null)
                    depthMapDisplay.material = depthVisualizationMaterial;
            }

            OnDepthUpdated?.Invoke(depthTexture);
        }
    }

    /// <summary>
    /// Gets physical metric distance (in meters) at a specific screen coordinate (pixels).
    /// Uses ARCore's low-latency CPU depth image readback.
    /// </summary>
    /// <param name="screenPosition">Screen coordinates in pixels (e.g., from Input or touch)</param>
    /// <param name="depthInMeters">Output depth in meters (0 if invalid/failed)</param>
    /// <returns>True if depth was successfully sampled, false otherwise</returns>
    public bool TryGetDepthAtScreenPoint(Vector2 screenPosition, out float depthInMeters)
    {
        depthInMeters = 0f;
        if (m_OcclusionManager == null) return false;

        if (!m_OcclusionManager.TryAcquireEnvironmentDepthCpuImage(out XRCpuImage depthImage))
            return false;

        using (depthImage)
        {
            // Normalize coordinates [0, 1]
            float u = Mathf.Clamp01(screenPosition.x / Screen.width);
            float v = Mathf.Clamp01(screenPosition.y / Screen.height);

            // Convert to image pixel coordinates
            int x = Mathf.Clamp((int)(u * depthImage.width), 0, depthImage.width - 1);
            int y = Mathf.Clamp((int)(v * depthImage.height), 0, depthImage.height - 1);

            var conversionParams = new XRCpuImage.ConversionParams
            {
                inputRect = new RectInt(x, y, 1, 1),
                outputDimensions = new Vector2Int(1, 1),
                outputFormat = TextureFormat.RFloat,
                transformation = XRCpuImage.Transformation.None
            };

            int bufferSize = depthImage.GetConvertedDataSize(conversionParams);
            var buffer = new NativeArray<byte>(bufferSize, Allocator.Temp);
            try
            {
                depthImage.Convert(conversionParams, buffer.Slice());
                var floatData = buffer.Reinterpret<float>(1);
                depthInMeters = floatData[0];
                return depthInMeters > 0f;
            }
            finally
            {
                buffer.Dispose();
            }
        }
    }

    /// <summary>
    /// Unprojects a 2D screen coordinate into a 3D world-space coordinate using physical ARCore depth.
    /// </summary>
    public bool TryGetWorldPositionFromDepth(Vector2 screenPosition, out Vector3 worldPosition)
    {
        worldPosition = Vector3.zero;

        if (!TryGetDepthAtScreenPoint(screenPosition, out float depthInMeters))
            return false;

        if (m_ARCamera == null)
            m_ARCamera = Camera.main;

        if (m_ARCamera == null)
            return false;

        Vector3 screenPointWithDepth = new Vector3(screenPosition.x, screenPosition.y, depthInMeters);
        worldPosition = m_ARCamera.ScreenToWorldPoint(screenPointWithDepth);
        return true;
    }
}
