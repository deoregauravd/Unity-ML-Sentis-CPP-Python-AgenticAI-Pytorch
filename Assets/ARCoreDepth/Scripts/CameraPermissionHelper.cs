using System;
using System.Collections;
using UnityEngine;
#if UNITY_ANDROID
using UnityEngine.Android;
#endif

/// <summary>
/// Handles camera runtime permission requests on mobile devices (Android/iOS)
/// with event callbacks and status messages.
/// </summary>
public class CameraPermissionHelper : MonoBehaviour
{
    public event Action OnPermissionGranted;
    public event Action OnPermissionDenied;

    [Tooltip("Status text element to display permission progress")]
    public UnityEngine.UI.Text statusText;

    bool m_PermissionRequested;

    void Start()
    {
        // Permission request is triggered on-demand via UI button
    }

    public IEnumerator CheckAndRequestCameraPermission()
    {
        UpdateStatus("Checking camera permission...");

#if UNITY_ANDROID && !UNITY_EDITOR
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            UpdateStatus("Requesting camera permission...");
            Permission.RequestUserPermission(Permission.Camera);
            m_PermissionRequested = true;

            // Wait for user to respond
            float timeout = 10f;
            while (!Permission.HasUserAuthorizedPermission(Permission.Camera) && timeout > 0f)
            {
                yield return new WaitForSeconds(0.5f);
                timeout -= 0.5f;
            }
        }

        if (Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            UpdateStatus("Camera permission granted.");
            OnPermissionGranted?.Invoke();
        }
        else
        {
            UpdateStatus("Camera permission denied. Please grant permission in settings.");
            OnPermissionDenied?.Invoke();
        }
#elif UNITY_IOS && !UNITY_EDITOR
        yield return Application.RequestUserAuthorization(UserAuthorization.WebCam);
        if (Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            UpdateStatus("Camera permission granted.");
            OnPermissionGranted?.Invoke();
        }
        else
        {
            UpdateStatus("Camera permission denied.");
            OnPermissionDenied?.Invoke();
        }
#else
        // In Unity Editor or Standalone
        UpdateStatus("Running in Editor (Camera allowed)");
        yield return null;
        OnPermissionGranted?.Invoke();
#endif
    }

    void UpdateStatus(string message)
    {
        if (statusText != null)
            statusText.text = message;
        Debug.Log($"[CameraPermission] {message}");
    }
}
