using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Procedurally constructs and articulates a futuristic, 3D Holographic Sci-Fi VR Hand
/// driven in real-time by the 21 landmarks of the BlazeHand tracking pipeline.
/// Features glowing fresnel rim joints, tapered finger bone cylinders, and a procedural palm plate.
/// </summary>
public class VRHolographicHand : MonoBehaviour
{
    [Header("Visual Styling")]
    [Tooltip("Hologram base color")]
    public Color hologramColor = new Color(0.1f, 0.85f, 1f, 0.8f);

    [Tooltip("Glow color at grazing angles (Fresnel rim)")]
    public Color rimGlowColor = new Color(0.4f, 0.95f, 1f, 1f);

    [Range(0.5f, 6f)]
    public float rimPower = 2.5f;

    [Header("Smoothing")]
    [Range(5f, 30f)]
    public float trackingSmoothing = 18f;

    // Constants
    const int k_NumKeypoints = 21;

    static readonly (int, int)[] HandBones = new (int, int)[]
    {
        (0, 1), (1, 2), (2, 3), (3, 4),        // Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        // Index
        (5, 9), (9, 10), (10, 11), (11, 12),   // Middle
        (9, 13), (13, 14), (14, 15), (15, 16), // Ring
        (13, 17), (17, 18), (18, 19), (19, 20),// Pinky
        (0, 17)                                // Palm edge
    };

    // Shared Material & Meshes
    Material m_HoloMaterial;
    Mesh m_SphereMesh;
    Mesh m_CylinderMesh;

    // 3D Nodes
    Transform[] m_JointNodes = new Transform[k_NumKeypoints];
    Transform[] m_BoneNodes = new Transform[HandBones.Length];
    Vector3[] m_TargetPositions = new Vector3[k_NumKeypoints];
    Vector3[] m_CurrentPositions = new Vector3[k_NumKeypoints];

    // Palm Mesh
    GameObject m_PalmObject;
    Mesh m_PalmMesh;
    Vector3[] m_PalmVertices = new Vector3[5]; // 0: Wrist, 1: Index, 2: Middle, 3: Ring, 4: Pinky

    // Root container
    GameObject m_VisualsRoot;

    bool m_IsHandVisible = false;
    float m_CurrentAlpha = 0f;

    void Awake()
    {
        CreateHologramMaterial();
        CreatePrimitiveMeshes();
        m_VisualsRoot = new GameObject("VisualsRoot");
        m_VisualsRoot.transform.SetParent(transform, false);
        BuildHandHierarchy();
        CreatePalmPlate();
    }

    void CreateHologramMaterial()
    {
        var shader = Shader.Find("SciFi/HolographicHand") 
                  ?? Shader.Find("Universal Render Pipeline/Unlit")
                  ?? Shader.Find("Unlit/Color");

        if (shader != null)
        {
            m_HoloMaterial = new Material(shader);
            m_HoloMaterial.SetColor("_HoloColor", hologramColor);
            m_HoloMaterial.SetColor("_RimColor", rimGlowColor);
            m_HoloMaterial.SetFloat("_RimPower", rimPower);
        }
    }

    void CreatePrimitiveMeshes()
    {
        // Extract shared meshes from temporary primitives to avoid asset clutter
        var tempSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        m_SphereMesh = tempSphere.GetComponent<MeshFilter>().sharedMesh;
        if (Application.isPlaying)
            Destroy(tempSphere);
        else
            DestroyImmediate(tempSphere);

        var tempCylinder = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        m_CylinderMesh = tempCylinder.GetComponent<MeshFilter>().sharedMesh;
        if (Application.isPlaying)
            Destroy(tempCylinder);
        else
            DestroyImmediate(tempCylinder);
    }

    void BuildHandHierarchy()
    {
        Transform parentTransform = m_VisualsRoot != null ? m_VisualsRoot.transform : transform;

        // 21 Joint Spheres
        for (int i = 0; i < k_NumKeypoints; i++)
        {
            var jointGo = new GameObject($"HoloJoint_{i}");
            jointGo.transform.SetParent(parentTransform, false);

            var mf = jointGo.AddComponent<MeshFilter>();
            mf.sharedMesh = m_SphereMesh;

            var mr = jointGo.AddComponent<MeshRenderer>();
            mr.sharedMaterial = m_HoloMaterial;
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;

            // Size nodes appropriately (tips are slightly smaller, knuckles slightly larger)
            float jointRadius = (i == 4 || i == 8 || i == 12 || i == 16 || i == 20) ? 0.012f : 0.016f;
            if (i == 0) jointRadius = 0.022f; // Wrist
            jointGo.transform.localScale = Vector3.one * jointRadius;

            m_JointNodes[i] = jointGo.transform;
            m_TargetPositions[i] = Vector3.zero;
            m_CurrentPositions[i] = Vector3.zero;
        }

        // 20 Bone Cylinders
        for (int b = 0; b < HandBones.Length; b++)
        {
            var boneGo = new GameObject($"HoloBone_{b}");
            boneGo.transform.SetParent(parentTransform, false);

            var mf = boneGo.AddComponent<MeshFilter>();
            mf.sharedMesh = m_CylinderMesh;

            var mr = boneGo.AddComponent<MeshRenderer>();
            mr.sharedMaterial = m_HoloMaterial;
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;

            m_BoneNodes[b] = boneGo.transform;
        }

        SetHandVisible(false);
    }

    void CreatePalmPlate()
    {
        Transform parentTransform = m_VisualsRoot != null ? m_VisualsRoot.transform : transform;
        m_PalmObject = new GameObject("HoloPalmPlate");
        m_PalmObject.transform.SetParent(parentTransform, false);

        var mf = m_PalmObject.AddComponent<MeshFilter>();
        var mr = m_PalmObject.AddComponent<MeshRenderer>();
        mr.sharedMaterial = m_HoloMaterial;
        mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;

        m_PalmMesh = new Mesh();
        m_PalmMesh.name = "PalmPlateMesh";

        // 5 vertices: 0 = Wrist, 1 = Index Knuckle(5), 2 = Middle Knuckle(9), 3 = Ring Knuckle(13), 4 = Pinky Knuckle(17)
        int[] triangles = new int[]
        {
            0, 1, 2,  // Wrist - Index - Middle
            0, 2, 3,  // Wrist - Middle - Ring
            0, 3, 4,  // Wrist - Ring - Pinky
            0, 2, 1,  // Double-sided backfaces
            0, 3, 2,
            0, 4, 3
        };

        m_PalmMesh.vertices = new Vector3[5];
        m_PalmMesh.triangles = triangles;
        m_PalmMesh.normals = new Vector3[5] { Vector3.back, Vector3.back, Vector3.back, Vector3.back, Vector3.back };
        mf.sharedMesh = m_PalmMesh;
    }

    public void SetHandVisible(bool visible)
    {
        m_IsHandVisible = visible;
        if (m_VisualsRoot != null)
            m_VisualsRoot.SetActive(visible);
    }

    /// <summary>
    /// Updates the 3D VR hand with the 21 world-space landmarks.
    /// </summary>
    public void UpdatePose(Vector3[] worldPositions, Camera referenceCamera = null)
    {
        if (worldPositions == null || worldPositions.Length < k_NumKeypoints)
            return;

        if (!m_IsHandVisible)
        {
            SetHandVisible(true);
            for (int i = 0; i < k_NumKeypoints; i++)
            {
                m_CurrentPositions[i] = worldPositions[i];
            }
        }

        for (int i = 0; i < k_NumKeypoints; i++)
        {
            m_TargetPositions[i] = worldPositions[i];
            m_CurrentPositions[i] = Vector3.Lerp(m_CurrentPositions[i], m_TargetPositions[i], Time.deltaTime * trackingSmoothing);

            if (m_JointNodes[i] != null)
            {
                m_JointNodes[i].position = m_CurrentPositions[i];
            }
        }

        Vector3 camUp = referenceCamera != null ? referenceCamera.transform.up : Vector3.up;

        // Position, rotate, and scale each bone cylinder between joints
        for (int b = 0; b < HandBones.Length; b++)
        {
            var (startIdx, endIdx) = HandBones[b];
            Vector3 pA = m_CurrentPositions[startIdx];
            Vector3 pB = m_CurrentPositions[endIdx];

            Vector3 delta = pB - pA;
            float length = delta.magnitude;

            if (length > 0.0001f && m_BoneNodes[b] != null)
            {
                m_BoneNodes[b].position = (pA + pB) * 0.5f;
                // Unity primitive cylinder is aligned along Y axis and height is 2 units
                m_BoneNodes[b].rotation = Quaternion.FromToRotation(Vector3.up, delta.normalized);
                float radius = (startIdx == 0) ? 0.009f : 0.007f;
                m_BoneNodes[b].localScale = new Vector3(radius, length * 0.5f, radius);
            }
        }

        // Update Palm Plate vertices
        if (m_PalmMesh != null)
        {
            m_PalmVertices[0] = m_PalmObject.transform.InverseTransformPoint(m_CurrentPositions[0]);  // Wrist
            m_PalmVertices[1] = m_PalmObject.transform.InverseTransformPoint(m_CurrentPositions[5]);  // Index
            m_PalmVertices[2] = m_PalmObject.transform.InverseTransformPoint(m_CurrentPositions[9]);  // Middle
            m_PalmVertices[3] = m_PalmObject.transform.InverseTransformPoint(m_CurrentPositions[13]); // Ring
            m_PalmVertices[4] = m_PalmObject.transform.InverseTransformPoint(m_CurrentPositions[17]); // Pinky

            m_PalmMesh.vertices = m_PalmVertices;
            m_PalmMesh.RecalculateNormals();
            m_PalmMesh.RecalculateBounds();
        }
    }

    /// <summary>
    /// Changes the hologram theme color dynamically.
    /// </summary>
    public void SetThemeColor(Color holoColor, Color rimColor)
    {
        hologramColor = holoColor;
        rimGlowColor = rimColor;
        if (m_HoloMaterial != null)
        {
            m_HoloMaterial.SetColor("_HoloColor", holoColor);
            m_HoloMaterial.SetColor("_RimColor", rimColor);
        }
    }
}
