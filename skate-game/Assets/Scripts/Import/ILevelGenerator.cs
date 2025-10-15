using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Encapsulates tunable parameters for level generation. These values can be
/// exposed in the UI to allow players or designers to adjust the detail
/// and smoothness of generated tracks.
/// </summary>
public struct LevelGenOptions
{
    /// <summary>
    /// Maximum number of vertices allowed in the output polyline. Very long
    /// paths can degrade performance; this limit should be enforced by
    /// individual generator implementations.
    /// </summary>
    public int maxEdgePoints;

    /// <summary>
    /// Smoothness factor (0–1). Higher values produce smoother tracks but
    /// may over-simplify the terrain. How this value is used depends on
    /// the generator implementation.
    /// </summary>
    public float smooth;

    /// <summary>
    /// Detail factor (0–1). Higher values retain more detail from the
    /// original contour. A lower value aggressively simplifies the track.
    /// </summary>
    public float detail;

    /// <summary>
    /// Whether to force the generated path to be open (not closed). Closed
    /// paths produce loops which may not be desired for a side-scrolling
    /// game. This flag gives the generator control over the topology.
    /// </summary>
    public bool openTrack;
}

/// <summary>
/// Represents a single polyline in a level definition. Additional fields
/// such as color or thickness could be added here in the future.
/// </summary>
[System.Serializable]
public class Polyline
{
    public string name;
    public bool closed;
    public List<Vector2> points;
}

/// <summary>
/// Data structure used to transfer level geometry from an external generator
/// into Unity. It contains a list of polylines, a scale factor (to convert
/// pixel units into Unity world units), and spawn/goal positions.
/// </summary>
[System.Serializable]
public class LevelData
{
    public float scale;
    public List<Polyline> polylines;
    public Vector2 spawn;
    public Vector2 goal;
}

/// <summary>
/// The common interface for all level generators. Implementations may use
/// different algorithms or external tools to produce a LevelData instance
/// from a given source texture and option set.
/// </summary>
public interface ILevelGenerator
{
    /// <summary>
    /// Generate a level from a source image. The returned LevelData
    /// contains the polylines defining the track along with scaling
    /// information and spawn/goal points.
    /// </summary>
    /// <param name="sourceTex">The source texture captured from user input.</param>
    /// <param name="options">Generator-specific tuning parameters.</param>
    /// <returns>A populated LevelData instance, or null if generation fails.</returns>
    LevelData Generate(Texture2D sourceTex, LevelGenOptions options);
}