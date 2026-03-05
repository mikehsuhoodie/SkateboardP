using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// A placeholder implementation of ILevelGenerator that runs entirely in
/// C#. To produce usable tracks, this class should implement an image
/// processing pipeline similar to the Python script (e.g., using marching
/// squares to extract contours and Chaikin smoothing). At present, it
/// returns null and logs a warning. See PythonGenBridge for a working
/// generator that invokes the Python script.
/// </summary>
public class BuiltInGenerator : ILevelGenerator
{
    public LevelData Generate(Texture2D sourceTex, LevelGenOptions options)
    {
        Debug.LogWarning("BuiltInGenerator is not implemented. Please select the Python generator.");
        // Returning null signals failure; calling code should handle this case.
        return null;
    }
}