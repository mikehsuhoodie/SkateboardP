using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// Captures or selects photos, sends them to the server, polls the generated result,
/// and enqueues each returned level into LevelCoordinator.
/// Supports a batch flow for selecting multiple gallery images at once.
/// </summary>
public class MobileCameraCapture : MonoBehaviour
{
    [System.Serializable]
    public class ServerResponse
    {
        public string status;
        public string foreground_base64;
        public string gameplay_base64;
        public string background_base64;
        public TrackData metadata;
    }

    [Header("Network Settings")]
    [Tooltip("Used when there is no saved IP in PlayerPrefs.")]
    public string defaultServerIP = "192.168.1.100";
    public string port = "5000";
    private string apiKey = "112550191";

    [Header("Polling Settings")]
    public int maxPollingAttempts = 5;
    public float pollingInterval = 1.0f;

    [Header("UI References")]
    public TMP_InputField ipInputField;
    public Button saveIPButton;

    [Header("Image Settings")]
    public int targetLongSide = 1920;
    [Range(1, 100)]
    public int jpgQuality = 85;

    [Header("Gallery Selection")]
    [Tooltip("When supported on the current mobile platform, the gallery button opens in multi-select mode.")]
    public bool preferMultipleGallerySelection = true;
    [Tooltip("If multi-select is unavailable, fall back to the original single-image picker.")]
    public bool fallbackToSingleGallerySelection = true;

    [Header("Architecture")]
    public LevelCoordinator levelCoordinator;

    [Header("UI Panels")]
    public GameObject uploadPanel;
    public Canvas controlCanva;
    public Canvas resultCanvas;

    private bool isProcessing;
    private bool lastUploadSuccessful;
    private string currentServerIP;

    private string GetSanitizedIP()
    {
        return currentServerIP?
            .Trim()
            .Replace(" ", string.Empty)
            .Replace("\n", string.Empty)
            .Replace("\r", string.Empty)
            .Replace("http://", string.Empty)
            .Replace("https://", string.Empty);
    }

    private string GetSanitizedPort()
    {
        return port?
            .Trim()
            .Replace(" ", string.Empty)
            .Replace("\n", string.Empty)
            .Replace("\r", string.Empty);
    }

    private string UploadUrl => $"http://{GetSanitizedIP()}:{GetSanitizedPort()}/upload";
    private string DownloadUrl => $"http://{GetSanitizedIP()}:{GetSanitizedPort()}/download";

    private void Start()
    {
        currentServerIP = PlayerPrefs.GetString("SavedServerIP", defaultServerIP);
        currentServerIP = GetSanitizedIP();

        if (ipInputField != null)
        {
            ipInputField.text = currentServerIP;
            ipInputField.onEndEdit.AddListener(UpdateIP);
        }

        if (saveIPButton != null)
        {
            saveIPButton.onClick.AddListener(() => UpdateIP(ipInputField != null ? ipInputField.text : currentServerIP));
        }

        Debug.Log($"[MobileCameraCapture] Active upload endpoint: {UploadUrl}");
    }

    private void Update()
    {
#if ENABLE_INPUT_SYSTEM
        if (Keyboard.current != null && Keyboard.current.cKey.wasPressedThisFrame && !isProcessing)
        {
            PickFromCamera();
        }
#endif
    }

    public void UpdateIP(string newIP)
    {
        if (string.IsNullOrEmpty(newIP))
        {
            return;
        }

        string sanitized = newIP
            .Trim()
            .Replace(" ", string.Empty)
            .Replace("\n", string.Empty)
            .Replace("\r", string.Empty);

        if (sanitized.StartsWith("http://"))
        {
            sanitized = sanitized.Replace("http://", string.Empty);
        }

        if (sanitized.StartsWith("https://"))
        {
            sanitized = sanitized.Replace("https://", string.Empty);
        }

        currentServerIP = sanitized;

        if (ipInputField != null)
        {
            ipInputField.text = currentServerIP;
        }

        PlayerPrefs.SetString("SavedServerIP", currentServerIP);
        PlayerPrefs.Save();
        Debug.Log($"[MobileCameraCapture] Saved server IP: {currentServerIP}");
    }

    private void ToggleGameUI()
    {
        if (uploadPanel != null) uploadPanel.SetActive(false);
        if (resultCanvas != null) resultCanvas.gameObject.SetActive(true);
        if (controlCanva != null) controlCanva.gameObject.SetActive(true);
    }

    public void PickFromCamera()
    {
        if (isProcessing)
        {
            return;
        }

        NativeCamera.TakePicture(path =>
        {
            if (!string.IsNullOrEmpty(path))
            {
                StartCoroutine(ProcessSelectedFiles(new[] { path }));
            }
        });
    }

    public void PickFromGallery()
    {
        if (isProcessing)
        {
            return;
        }

        if (preferMultipleGallerySelection && NativeGallery.CanSelectMultipleFilesFromGallery())
        {
            PickMultipleFromGallery();
            return;
        }

        if (preferMultipleGallerySelection && !fallbackToSingleGallerySelection)
        {
            Debug.LogWarning("[MobileCameraCapture] Multi-select gallery is unavailable on this platform/device.");
            return;
        }

        PickSingleFromGallery();
    }

    public void PickSingleFromGallery()
    {
        if (isProcessing)
        {
            return;
        }

        NativeGallery.GetImageFromGallery(path =>
        {
            if (!string.IsNullOrEmpty(path))
            {
                StartCoroutine(ProcessSelectedFiles(new[] { path }));
            }
        });
    }

    public void PickMultipleFromGallery()
    {
        if (isProcessing)
        {
            return;
        }

        if (!NativeGallery.CanSelectMultipleFilesFromGallery())
        {
            if (fallbackToSingleGallerySelection)
            {
                PickSingleFromGallery();
            }
            else
            {
                Debug.LogWarning("[MobileCameraCapture] Multi-select gallery is unavailable on this platform/device.");
            }

            return;
        }

        NativeGallery.GetImagesFromGallery(paths =>
        {
            if (paths != null && paths.Length > 0)
            {
                StartCoroutine(ProcessSelectedFiles(paths));
            }
        });
    }

    private IEnumerator ProcessSelectedFiles(IReadOnlyList<string> paths)
    {
        if (paths == null || paths.Count == 0)
        {
            yield break;
        }

        isProcessing = true;

        int total = 0;
        for (int i = 0; i < paths.Count; i++)
        {
            if (!string.IsNullOrEmpty(paths[i]))
            {
                total++;
            }
        }

        if (total == 0)
        {
            isProcessing = false;
            yield break;
        }

        int current = 0;
        for (int i = 0; i < paths.Count; i++)
        {
            string path = paths[i];
            if (string.IsNullOrEmpty(path))
            {
                continue;
            }

            current++;
            yield return StartCoroutine(ProcessSingleSelectedFile(path, current, total));
        }

        isProcessing = false;
    }

    private IEnumerator ProcessSingleSelectedFile(string path, int index, int total)
    {
        Debug.Log($"[MobileCameraCapture] Processing image {index}/{total}: {path}");

        Texture2D original = NativeCamera.LoadImageAtPath(path, -1, false);
        if (original == null)
        {
            Debug.LogError($"[MobileCameraCapture] Failed to load image: {path}");
            yield break;
        }

        Texture2D resized = GetResizedTexture(original);
        byte[] jpgData = resized.EncodeToJPG(jpgQuality);

        Destroy(original);
        if (resized != original)
        {
            Destroy(resized);
        }

        yield return StartCoroutine(PostImageToServer(jpgData, index, total));

        if (lastUploadSuccessful)
        {
            yield return StartCoroutine(DownloadImageInternal(index, total));
        }
    }

    private Texture2D GetResizedTexture(Texture2D source)
    {
        if (source == null)
        {
            return null;
        }

        int sourceWidth = source.width;
        int sourceHeight = source.height;

        int targetWidth;
        int targetHeight;

        if (sourceWidth >= sourceHeight)
        {
            targetWidth = targetLongSide;
            targetHeight = Mathf.RoundToInt((float)sourceHeight / sourceWidth * targetLongSide);
        }
        else
        {
            targetHeight = targetLongSide;
            targetWidth = Mathf.RoundToInt((float)sourceWidth / sourceHeight * targetLongSide);
        }

        Debug.Log($"[MobileCameraCapture] Resize {sourceWidth}x{sourceHeight} -> {targetWidth}x{targetHeight}");
        return Resize(source, targetWidth, targetHeight);
    }

    private Texture2D Resize(Texture2D source, int targetWidth, int targetHeight)
    {
        RenderTexture rt = RenderTexture.GetTemporary(
            targetWidth,
            targetHeight,
            0,
            RenderTextureFormat.Default,
            RenderTextureReadWrite.Linear);

        Graphics.Blit(source, rt);

        RenderTexture previous = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D result = new Texture2D(targetWidth, targetHeight, TextureFormat.RGB24, false);
        result.ReadPixels(new Rect(0, 0, targetWidth, targetHeight), 0, 0);
        result.Apply();

        RenderTexture.active = previous;
        RenderTexture.ReleaseTemporary(rt);
        return result;
    }

    private IEnumerator PostImageToServer(byte[] data, int index, int total)
    {
        WWWForm form = new WWWForm();
        form.AddBinaryData("photo", data, $"mobile_upload_{index:D2}.jpg", "image/jpeg");

        using (UnityWebRequest www = UnityWebRequest.Post(UploadUrl, form))
        {
            www.SetRequestHeader("X-API-KEY", apiKey);
            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[MobileCameraCapture] Upload failed for image {index}/{total}: {www.error}");
                lastUploadSuccessful = false;
            }
            else
            {
                Debug.Log($"[MobileCameraCapture] Upload succeeded for image {index}/{total}.");
                lastUploadSuccessful = true;
            }
        }
    }

    public void GetImageFromServer()
    {
        if (isProcessing)
        {
            return;
        }

        StartCoroutine(DownloadImageRoutine());
    }

    private IEnumerator DownloadImageRoutine()
    {
        isProcessing = true;
        yield return StartCoroutine(DownloadImageInternal(1, 1));
        isProcessing = false;
    }

    private IEnumerator DownloadImageInternal(int batchIndex, int batchTotal)
    {
        int attempts = 0;
        bool isDone = false;

        while (attempts < maxPollingAttempts && !isDone)
        {
            attempts++;
            Debug.Log($"[MobileCameraCapture] Poll {batchIndex}/{batchTotal}, attempt {attempts}/{maxPollingAttempts}: {DownloadUrl}");

            using (UnityWebRequest www = UnityWebRequest.Get(DownloadUrl))
            {
                www.SetRequestHeader("X-API-KEY", apiKey);
                yield return www.SendWebRequest();

                if (www.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"[MobileCameraCapture] Download failed for image {batchIndex}/{batchTotal}: {www.error}");
                }
                else
                {
                    ServerResponse response = null;

                    try
                    {
                        response = JsonConvert.DeserializeObject<ServerResponse>(www.downloadHandler.text);
                    }
                    catch (System.Exception ex)
                    {
                        Debug.LogError($"[MobileCameraCapture] Failed to parse server response: {ex.Message}");
                        isDone = true;
                    }

                    if (response != null)
                    {
                        if (response.status == "processing")
                        {
                            Debug.Log($"[MobileCameraCapture] Server still processing image {batchIndex}/{batchTotal}.");
                        }
                        else if (response.status == "success" || response.metadata != null)
                        {
                            QueueServerResponse(response, batchIndex);
                            ToggleGameUI();
                            isDone = true;
                        }
                        else
                        {
                            Debug.LogError($"[MobileCameraCapture] Server returned error status: {response.status}");
                            isDone = true;
                        }
                    }
                }
            }

            if (!isDone && attempts < maxPollingAttempts)
            {
                yield return new WaitForSeconds(pollingInterval);
            }
        }

        if (!isDone)
        {
            Debug.LogError($"[MobileCameraCapture] Polling timed out after {maxPollingAttempts} attempts for image {batchIndex}/{batchTotal}.");
        }
    }

    private void QueueServerResponse(ServerResponse response, int batchIndex)
    {
        if (response == null)
        {
            return;
        }

        if (response.metadata != null)
        {
            string trackJson = JsonConvert.SerializeObject(response.metadata);
            string savePath = Path.Combine(Application.persistentDataPath, $"schema_{batchIndex:D2}.json");
            File.WriteAllText(savePath, trackJson);
            File.WriteAllText(Path.Combine(Application.persistentDataPath, "schema.json"), trackJson);
            Debug.Log($"[MobileCameraCapture] Saved metadata backup: {savePath}");
        }

        LevelPayload payload = new LevelPayload
        {
            TrackData = response.metadata,
            ForegroundTexture = DecodeBase64ToTexture(response.foreground_base64),
            GameplayTexture = DecodeBase64ToTexture(response.gameplay_base64),
            BackgroundTexture = DecodeBase64ToTexture(response.background_base64)
        };

        if (levelCoordinator != null)
        {
            levelCoordinator.EnqueueLevel(payload);
        }
        else
        {
            Debug.LogWarning("[MobileCameraCapture] LevelCoordinator is not assigned, payload was not enqueued.");
        }
    }

    private Texture2D DecodeBase64ToTexture(string base64)
    {
        if (string.IsNullOrEmpty(base64))
        {
            return null;
        }

        try
        {
            byte[] imageBytes = System.Convert.FromBase64String(base64);
            Texture2D tex = new Texture2D(2, 2);
            if (tex.LoadImage(imageBytes))
            {
                return tex;
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[MobileCameraCapture] Error decoding base64 texture: {e.Message}");
        }

        return null;
    }
}
