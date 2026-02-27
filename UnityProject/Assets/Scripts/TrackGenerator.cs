using System;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using Newtonsoft.Json;

public class TrackGenerator : MonoBehaviour
{
    [Header("Settings")]
    public string schemaFileName = "schema.json";
    public float orthographicSize = 5.0f;
    
    [Header("Physics")]
    public EdgeCollider2D edgeCollider;

    [System.Serializable]
    public class TrackData
    {
        public string version;
        public string timestamp;
        public string source_image;
        public float aspect_ratio;
        public List<float[]> points;
        public List<int[]> occluded_segments;
        public string terrain_type;
        public float friction_coefficient;
    }

    void Start()
    {
        if (edgeCollider == null)
            edgeCollider = GetComponent<EdgeCollider2D>();

        LoadAndGenerate();
    }

    [ContextMenu("Generate Track")]
    public void LoadAndGenerate()
    {
        string filePath = Path.Combine(Application.dataPath, "../../", schemaFileName);
        if (!File.Exists(filePath))
        {
            Debug.LogError($"Schema file not found at: {filePath}");
            return;
        }

        string json = File.ReadAllText(filePath);
        TrackData data = JsonConvert.DeserializeObject<TrackData>(json);

        if (data != null && data.points != null)
        {
            GenerateCollider(data);
        }
    }

    void GenerateCollider(TrackData data)
    {
        Vector2[] worldPoints = new Vector2[data.points.Count];
        float aspect = data.aspect_ratio > 0 ? data.aspect_ratio : (float)Screen.width / Screen.height;
        float h = orthographicSize * 2.0f;
        float w = h * aspect;

        for (int i = 0; i < data.points.Count; i++)
        {
            float xn = data.points[i][0];
            float yn = data.points[i][1];

            // Mapping: (0,0) image bottom-left to Unity world
            float ux = (xn - 0.5f) * w;
            float uy = (yn - 0.5f) * h;
            worldPoints[i] = new Vector2(ux, uy);
        }

        edgeCollider.points = worldPoints;
        Debug.Log($"Generated EdgeCollider with {worldPoints.Length} points.");
    }
}
