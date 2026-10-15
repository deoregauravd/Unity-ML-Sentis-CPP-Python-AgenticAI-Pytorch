using UnityEngine;

public class FacePreview : MonoBehaviour
{
    public BoundingBox boundingBox;
    public Keypoint[] keypoints;

    public void SetActive(bool active)
    {
        gameObject.SetActive(active);
    }

    public void SetBoundingBox(bool active, Vector3 position, Vector2 size)
    {
        if (boundingBox != null)
            boundingBox.Set(active, position, size);
    }

    public void SetKeypoint(int index, bool active, Vector3 position)
    {
        if (keypoints != null && index >= 0 && index < keypoints.Length && keypoints[index] != null)
        {
            keypoints[index].Set(active, position);
        }
    }
}
