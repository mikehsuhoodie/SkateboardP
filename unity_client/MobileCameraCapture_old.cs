using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI; 
using TMPro;
using System.Collections;
using System.IO;
using Newtonsoft.Json;

#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

/// <summary>
/// 注意：此腳本需要安裝 NativeCamera 與 NativeGallery 插件。
/// GitHub: https://github.com/yasirkula/UnityNativeCamera
/// GitHub: https://github.com/yasirkula/UnityNativeGallery
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
    [Tooltip("如果未填寫 ipInputField，則預設使用此 IP")]
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
    public int targetLongSide = 1920; // Ensure 1080p-class resolution
    [Range(1, 100)]
    public int jpgQuality = 85;

    [Header("Architecture")]
    public LevelCoordinator levelCoordinator;

    [Header("UI Panels")]
    public GameObject uploadPanel;    // 上傳介面 (預設應在 Overlay Canvas)
    public Canvas controlCanva;
    public Canvas resultCanvas;       // 專門用來顯示結果圖的 UI Canvas

    private bool isProcessing = false;
    private bool lastUploadSuccessful = false;
    private string currentServerIP;

    // 動態取得網址 (加入更強健的判斷)
    private string GetSanitizedIP() => currentServerIP?.Trim().Replace(" ", "").Replace("\n", "").Replace("\r", "").Replace("http://", "").Replace("https://", "");
    private string GetSanitizedPort() => port?.Trim().Replace(" ", "").Replace("\n", "").Replace("\r", "");

    private string UploadUrl => $"http://{GetSanitizedIP()}:{GetSanitizedPort()}/upload";
    private string DownloadUrl => $"http://{GetSanitizedIP()}:{GetSanitizedPort()}/download";

    void Start()
    {
        // 從 PlayerPrefs 讀取上次儲存的 IP，如果沒有就用預設值
        currentServerIP = PlayerPrefs.GetString("SavedServerIP", defaultServerIP);
        
        // 初次執行也進行一次清理
        currentServerIP = GetSanitizedIP();

        // 2. 如果有設定 InputField，將其文字初始化
        if (ipInputField != null)
        {
            ipInputField.text = currentServerIP;
            ipInputField.onEndEdit.AddListener(UpdateIP);
        }

        if (saveIPButton != null)
        {
            saveIPButton.onClick.AddListener(() => UpdateIP(ipInputField.text));
        }

        Debug.Log($"[ServerSettings] 當前伺服器網址: {UploadUrl}");
    }

    // 更新 IP 的方法 (加入清道夫邏輯)
    public void UpdateIP(string newIP)
    {
        if (string.IsNullOrEmpty(newIP)) return;

        // 清理輸入內容：去空白、去換行、去 http 前綴
        string sanitized = newIP.Trim().Replace(" ", "").Replace("\n", "").Replace("\r", "");
        if (sanitized.StartsWith("http://")) sanitized = sanitized.Replace("http://", "");
        if (sanitized.StartsWith("https://")) sanitized = sanitized.Replace("https://", "");

        currentServerIP = sanitized;
        
        // 同步回 UI
        if (ipInputField != null) ipInputField.text = currentServerIP;

        PlayerPrefs.SetString("SavedServerIP", currentServerIP);
        PlayerPrefs.Save();
        Debug.Log($"[ServerSettings] IP 已更新並儲存: {currentServerIP} (完整網址: {UploadUrl})");
    }

    void Update()
    {
        #if ENABLE_INPUT_SYSTEM
        if (Keyboard.current != null)
        {
            if (Keyboard.current.cKey.wasPressedThisFrame && !isProcessing)
            {
                PickFromCamera(); 
            }
        }
        #endif
    }

    private void ToggleGameUI()
    {
        if (uploadPanel != null) uploadPanel.SetActive(false);
        if (resultCanvas != null) resultCanvas.gameObject.SetActive(true);
        if (controlCanva != null) controlCanva.gameObject.SetActive(true);
    }

    // --- Public 介面 (可連結 UI Button) ---

    public void PickFromCamera()
    {
        if (isProcessing) return;

        // 呼叫原生相機
        NativeCamera.TakePicture((path) =>
        {
            if (path != null)
            {
                StartCoroutine(ProcessAndSendFile(path));
            }
        });
    }

    public void PickFromGallery()
    {
        if (isProcessing) return;

        // 呼叫原生圖庫
        NativeGallery.GetImageFromGallery((path) =>
        {
            if (path != null)
            {
                StartCoroutine(ProcessAndSendFile(path));
            }
        });
    }

    // --- 核心處理邏輯 ---

    private IEnumerator ProcessAndSendFile(string path)
    {
        isProcessing = true;

        // 1. 從路徑讀取圖片為 Texture2D
        Texture2D original = NativeCamera.LoadImageAtPath(path, -1, false); // -1 是讀取原始解析度
        if (original == null)
        {
            Debug.LogError("無法讀取圖片: " + path);
            isProcessing = false;
            yield break;
        }

        // 2. 執行縮放
        Texture2D resized = GetResizedTexture(original);
        
        // 3. 轉成 JPG 位元組
        byte[] jpgData = resized.EncodeToJPG(jpgQuality);

        // 釋放記憶體
        Destroy(original);
        if (resized != original) Destroy(resized);

        // 4. 發送至電腦
        yield return StartCoroutine(PostImageToServer(jpgData));

        // 5. 如果上傳成功，則自動開始下載（連同軌道生成）
        if (lastUploadSuccessful)
        {
            yield return StartCoroutine(DownloadImage());
        }

        isProcessing = false;
    }

    /// <summary>
    /// 根據 targetLongSide 計算縮放比例並返回縮放後的貼圖
    /// </summary>
    private Texture2D GetResizedTexture(Texture2D source)
    {
        if (source == null) return null;

        int sourceWidth = source.width;
        int sourceHeight = source.height;
        int targetWidth, targetHeight;

        if (sourceWidth >= sourceHeight) // 橫向或正方
        {
            targetWidth = targetLongSide;
            targetHeight = Mathf.RoundToInt((float)sourceHeight / sourceWidth * targetLongSide);
        }
        else // 縱向
        {
            targetHeight = targetLongSide;
            targetWidth = Mathf.RoundToInt((float)sourceWidth / sourceHeight * targetLongSide);
        }

        Debug.Log($"[Resize] {sourceWidth}x{sourceHeight} -> {targetWidth}x{targetHeight}");
        return Resize(source, targetWidth, targetHeight);
    }

    private Texture2D Resize(Texture2D source, int targetWidth, int targetHeight)
    {
        RenderTexture rt = RenderTexture.GetTemporary(targetWidth, targetHeight, 0, RenderTextureFormat.Default, RenderTextureReadWrite.Linear);
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

    private IEnumerator PostImageToServer(byte[] data)
    {
        WWWForm form = new WWWForm();
        form.AddBinaryData("photo", data, "mobile_upload.jpg", "image/jpeg");

        using (UnityWebRequest www = UnityWebRequest.Post(UploadUrl, form))
        {
            www.SetRequestHeader("X-API-KEY", apiKey);
            yield return www.SendWebRequest();

            if (www.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"傳送到電腦失敗 ({UploadUrl}): " + www.error);
                lastUploadSuccessful = false;
            }
            else
            {
                Debug.Log("圖片已成功傳送到電腦！");
                lastUploadSuccessful = true;
            }
        }
    }

    /// <summary>
    /// 從伺服器下載圖片並顯示在 RawImage 上
    /// </summary>
    public void GetImageFromServer()
    {
        if (isProcessing) return;
        StartCoroutine(DownloadImage());
    }

    private IEnumerator DownloadImage()
    {
        isProcessing = true;
        int attempts = 0;
        bool isDone = false;

        while (attempts < maxPollingAttempts && !isDone)
        {
            attempts++;
            Debug.Log($"正在下載 JSON 數據 (嘗試 {attempts}/{maxPollingAttempts}): {DownloadUrl}");

            using (UnityWebRequest www = UnityWebRequest.Get(DownloadUrl))
            {
                www.SetRequestHeader("X-API-KEY", apiKey);
                yield return www.SendWebRequest();

                if (www.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"下載暫時失敗 ({www.error})，{pollingInterval} 秒後重試...");
                }
                else
                {
                    try
                    {
                        string jsonString = www.downloadHandler.text;
                        ServerResponse response = JsonConvert.DeserializeObject<ServerResponse>(jsonString);

                        if (response != null)
                        {
                            // 檢查伺服器狀態
                            if (response.status == "processing")
                            {
                                Debug.Log("伺服器還在處理中，等待重試...");
                            }
                            else if (response.status == "success" || response.metadata != null)
                            {
                                // 成功拿到資料
                                if (response.metadata != null)
                                {
                                    string trackJson = JsonConvert.SerializeObject(response.metadata);
                                    string savePath = Path.Combine(Application.persistentDataPath, "schema.json");
                                    File.WriteAllText(savePath, trackJson);
                                    Debug.Log($"[Backup] 已將軌道資料儲存至: {savePath}");
                                }

                                LevelPayload payload = new LevelPayload();
                                payload.TrackData = response.metadata;
                                payload.ForegroundTexture = DecodeBase64ToTexture(response.foreground_base64);
                                payload.GameplayTexture = DecodeBase64ToTexture(response.gameplay_base64);
                                payload.BackgroundTexture = DecodeBase64ToTexture(response.background_base64);

                                if (levelCoordinator != null)
                                {
                                    levelCoordinator.ApplyPayload(payload);
                                }

                                ToggleGameUI();
                                isDone = true; // 成功獲取，結束迴圈
                            }
                            else
                            {
                                Debug.LogError($"伺服器回傳未知狀態: {response.status}");
                                isDone = true; // 發生未知錯誤，停止重試
                            }
                        }
                    }
                    catch (System.Exception ex)
                    {
                        Debug.LogError("解析回應時發生錯誤: " + ex.Message);
                        isDone = true; // 解析失敗通常不需要重試
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
            Debug.LogError($"已達到最大重試次數 ({maxPollingAttempts})，下載結束。");
        }

        isProcessing = false;
    }

    private Texture2D DecodeBase64ToTexture(string base64)
    {
        if (string.IsNullOrEmpty(base64)) return null;
        try
        {
            byte[] imageBytes = System.Convert.FromBase64String(base64);
            Texture2D tex = new Texture2D(2, 2);
            if (tex.LoadImage(imageBytes)) return tex;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error decoding base64 texture: {e.Message}");
        }
        return null;
    }
}
