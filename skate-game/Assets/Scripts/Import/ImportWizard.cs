using System.IO;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// A simple UI controller for selecting an image file, choosing a level
/// generator, and producing a LevelData instance. This script is designed
/// for use in the Unity Editor or a desktop build and is not meant for
/// mobile platforms (which should provide their own image picker).
/// </summary>
public class ImportWizard : MonoBehaviour
{
    [Header("UI References")]
    public RawImage preview;            // Displays a thumbnail of the selected image
    public Dropdown generatorDropdown;  // Allows the user to pick between available generators

    // Internal state
    private Texture2D sourceTex;
    private ILevelGenerator generator;

    /// <summary>
    /// Read an image file from disk and display it. This method can be
    /// called from a file picker callback or a debug menu in the Editor.
    /// </summary>
    /// <param name="path">Absolute path to the image file.</param>
    public void OnPickImageFromPath(string path)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            Debug.LogError($"ImportWizard: invalid image path: {path}");
            return;
        }
        byte[] data = File.ReadAllBytes(path);
        Texture2D tex = new Texture2D(2, 2);
        if (!tex.LoadImage(data))
        {
            Debug.LogError("ImportWizard: failed to load image data");
            return;
        }
        sourceTex = tex;
        if (preview != null)
        {
            preview.texture = sourceTex;
        }
    }

    /// <summary>
    /// Assign a Texture2D directly (for example from a webcam). The preview
    /// display will be updated if present.
    /// </summary>
    /// <param name="tex">Texture to use as the source image.</param>
    public void OnPickTexture(Texture2D tex)
    {
        sourceTex = tex;
        if (preview != null)
        {
            preview.texture = sourceTex;
        }
    }

    /// <summary>
    /// Called when the generator selection changes. Configures the
    /// appropriate ILevelGenerator implementation. Dropdown values:
    /// 0 = BuiltInGenerator, 1 = PythonGenBridge.
    /// </summary>
    public void OnGeneratorChanged()
    {
        switch (generatorDropdown != null ? generatorDropdown.value : 0)
        {
            case 0:
                generator = new BuiltInGenerator();
                break;
            case 1:
                generator = new PythonGenBridge();
                break;
            default:
                generator = new BuiltInGenerator();
                break;
        }
    }

    /// <summary>
    /// Generate a level immediately using the selected generator and
    /// options. The caller is responsible for providing LevelGenOptions
    /// appropriate to the generator. If generation fails, returns null.
    /// </summary>
    /// <param name="options">Generator tuning parameters.</param>
    /// <returns>The generated LevelData, or null on failure.</returns>
    public LevelData GenerateNow(LevelGenOptions options)
    {
        if (sourceTex == null)
        {
            Debug.LogError("ImportWizard: no source image loaded");
            return null;
        }
        if (generator == null)
        {
            OnGeneratorChanged();
        }
        if (generator == null)
        {
            Debug.LogError("ImportWizard: generator is null");
            return null;
        }
        return generator.Generate(sourceTex, options);
    }
}