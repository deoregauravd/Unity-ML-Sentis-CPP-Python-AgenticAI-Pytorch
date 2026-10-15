using UnityEngine;

public class ImagePreview : MonoBehaviour
{
    public GameObject imageQuad;

    public void SetTexture(Texture texture)
    {
        if (texture == null) return;

        if (imageQuad == null)
            imageQuad = gameObject;

        var renderer = imageQuad.GetComponent<MeshRenderer>();
        if (renderer != null && renderer.material != null)
        {
            renderer.material.mainTexture = texture;
        }

        var webCam = texture as WebCamTexture;
        int rotAngle = webCam != null ? webCam.videoRotationAngle : 0;
        bool isMirrored = webCam != null && webCam.videoVerticallyMirrored;

        var aspectRatio = texture.width / (float)texture.height;
        if (rotAngle == 90 || rotAngle == 270)
        {
            aspectRatio = (float)texture.height / texture.width;
        }

        imageQuad.transform.localEulerAngles = new Vector3(0, 0, -rotAngle);
        imageQuad.transform.localScale = new Vector3(aspectRatio, isMirrored ? -1f : 1f, 1f);
    }
}
