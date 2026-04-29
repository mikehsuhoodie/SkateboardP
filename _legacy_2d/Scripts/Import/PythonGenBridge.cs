using System.Diagnostics;
using System.IO;
using UnityEngine;

/// <summary>
/// An implementation of ILevelGenerator that bridges Unity to an external
/// Python script. It writes a temporary PNG file for the source texture,
/// invokes the Python script as a separate process, waits for it to
/// complete, then reads the resulting JSON back into a LevelData object.
/// </summary>
public class PythonGenBridge : ILevelGenerator
{
    /// <summary>
    /// Path to the Python executable. Adjust this if your Python
    /// installation is named differently or lives in a non-standard
    /// location (e.g., "python3" on Linux/macOS).
    /// </summary>
    public string pythonPath = "python";

    /// <summary>
    /// Path to the level generation script. Should point to the
    /// level_gen.py file relative to the working directory or an
    /// absolute path. Use forward slashes or escape backslashes on
    /// Windows.
    /// </summary>
    public string scriptPath = "level-gen/level_gen.py";

    public LevelData Generate(Texture2D sourceTex, LevelGenOptions options)
    {
        if (sourceTex == null)
        {
            Debug.LogError("PythonGenBridge: source texture is null");
            return null;
        }

        // Prepare temporary files in Unity's persistent or temporary cache
        string tempDir = Application.temporaryCachePath;
        Directory.CreateDirectory(tempDir);
        string inputPath = Path.Combine(tempDir, "levelgen_input.png");
        string previewPath = Path.Combine(tempDir, "levelgen_preview.png");
        string jsonPath = Path.Combine(tempDir, "levelgen_output.json");

        // Write the source texture to disk
        byte[] pngData = sourceTex.EncodeToPNG();
        File.WriteAllBytes(inputPath, pngData);

        // Build command line arguments
        // On Windows, arguments must be quoted if paths contain spaces
        string args = $"\"{scriptPath}\" \"{inputPath}\" \"{previewPath}\" \"{jsonPath}\"";

        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = pythonPath,
            Arguments = args,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };
        try
        {
            using (Process process = Process.Start(startInfo))
            {
                // Optionally capture output for debugging
                string output = process.StandardOutput.ReadToEnd();
                string error = process.StandardError.ReadToEnd();
                process.WaitForExit();
                if (process.ExitCode != 0)
                {
                    Debug.LogError($"Python generator returned exit code {process.ExitCode}\n{error}");
                    return null;
                }
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Failed to start Python process: {ex}");
            return null;
        }

        // Read the JSON output
        if (!File.Exists(jsonPath))
        {
            Debug.LogError("Python generator did not produce a JSON file");
            return null;
        }
        string jsonText = File.ReadAllText(jsonPath);
        try
        {
            LevelData levelData = JsonUtility.FromJson<LevelData>(jsonText);
            return levelData;
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Failed to deserialize level JSON: {ex}");
            return null;
        }
    }
}