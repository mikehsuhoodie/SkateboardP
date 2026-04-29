using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Responsible for instantiating Unity colliders and game objects based on
/// LevelData generated from an external source. This class instantiates
/// track segments from a prefab containing an EdgeCollider2D and spawns
/// player and goal prefabs at the positions specified in the level data.
/// </summary>
public class LevelLoader : MonoBehaviour
{
    [Header("Prefab References")]
    public GameObject trackPrefab;
    public GameObject playerPrefab;
    public GameObject goalPrefab;

    // Keep track of objects spawned for the current level so they can be
    // cleaned up when loading a new level
    private readonly List<GameObject> spawnedTracks = new List<GameObject>();
    private GameObject spawnedPlayer;
    private GameObject spawnedGoal;

    /// <summary>
    /// Construct the level geometry and spawn entities. Existing level
    /// objects will be destroyed. If level data is null, the method
    /// exits silently.
    /// </summary>
    /// <param name="data">Level definition to instantiate.</param>
    public void LoadLevel(LevelData data)
    {
        if (data == null)
        {
            Debug.LogError("LevelLoader: no level data provided");
            return;
        }
        // Clear previous level objects
        ClearLevel();
        // Instantiate each polyline as an EdgeCollider2D
        foreach (var poly in data.polylines)
        {
            GameObject track = Instantiate(trackPrefab);
            var edge = track.GetComponent<EdgeCollider2D>();
            if (edge == null)
            {
                Debug.LogError("LevelLoader: track prefab lacks an EdgeCollider2D component");
                Destroy(track);
                continue;
            }
            // Transform points from pixel coordinates into Unity units by multiplying with scale
            var pts = new List<Vector2>();
            foreach (var p in poly.points)
            {
                pts.Add(p * data.scale);
            }
            edge.SetPoints(pts);
            spawnedTracks.Add(track);
        }
        // Spawn player and goal at their respective positions
        if (playerPrefab != null)
        {
            spawnedPlayer = Instantiate(playerPrefab, data.spawn * data.scale, Quaternion.identity);
        }
        if (goalPrefab != null)
        {
            spawnedGoal = Instantiate(goalPrefab, data.goal * data.scale, Quaternion.identity);
        }
    }

    /// <summary>
    /// Destroy all objects associated with the previously loaded level.
    /// </summary>
    public void ClearLevel()
    {
        foreach (var obj in spawnedTracks)
        {
            if (obj != null)
            {
                Destroy(obj);
            }
        }
        spawnedTracks.Clear();
        if (spawnedPlayer != null)
        {
            Destroy(spawnedPlayer);
            spawnedPlayer = null;
        }
        if (spawnedGoal != null)
        {
            Destroy(spawnedGoal);
            spawnedGoal = null;
        }
    }
}