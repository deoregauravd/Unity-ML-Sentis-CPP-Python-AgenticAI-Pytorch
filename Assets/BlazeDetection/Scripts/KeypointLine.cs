using UnityEngine;

public class KeypointLine : MonoBehaviour
{
    public LineRenderer lineRenderer;
    public Keypoint start;
    public Keypoint end;
    public Color color = Color.yellow;
    public float width = 0.005f;

    void Awake()
    {
        if (lineRenderer == null)
            lineRenderer = GetComponent<LineRenderer>();
    }

    void Start()
    {
        if (lineRenderer != null)
        {
            lineRenderer.startColor = color;
            lineRenderer.endColor = color;
            lineRenderer.startWidth = width;
            lineRenderer.endWidth = width;
            lineRenderer.positionCount = 2;
        }
    }

    void Update()
    {
        if (lineRenderer == null || start == null || end == null) return;

        bool active = start.IsActive && end.IsActive;
        lineRenderer.enabled = active;
        if (active)
        {
            lineRenderer.SetPosition(0, start.Position);
            lineRenderer.SetPosition(1, end.Position);
        }
    }
}
