using UnityEngine;

public class Keypoint : MonoBehaviour
{
    public LineRenderer outerCircle;
    public LineRenderer innerCircle;
    bool m_IsActive;
    Vector3 m_Position;

    public bool IsActive => m_IsActive;
    public Vector3 Position => m_Position;

    public Color outerColor = Color.white;
    public Color innerColor = Color.cyan;
    public float outerWidth = 0.02f;
    public float innerWidth = 0.01f;

    public void Start()
    {
        if (outerCircle != null)
        {
            outerCircle.startColor = outerColor;
            outerCircle.endColor = outerColor;
            outerCircle.startWidth = outerWidth;
            outerCircle.endWidth = outerWidth;
        }
        if (innerCircle != null)
        {
            innerCircle.startColor = innerColor;
            innerCircle.endColor = innerColor;
            innerCircle.startWidth = innerWidth;
            innerCircle.endWidth = innerWidth;
        }
    }

    public void Set(bool active, Vector3 position)
    {
        m_IsActive = active;
        m_Position = position;
        gameObject.SetActive(active);
        if (!active) return;

        if (outerCircle != null && outerCircle.positionCount >= 2)
        {
            outerCircle.SetPosition(0, position);
            outerCircle.SetPosition(1, position);
        }
        if (innerCircle != null && innerCircle.positionCount >= 2)
        {
            innerCircle.SetPosition(0, position);
            innerCircle.SetPosition(1, position);
        }
    }
}
