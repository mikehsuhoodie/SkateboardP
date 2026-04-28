# API Contract

## Current Response Format (infer.py)

The current `POST /infer` endpoint returns a JSON payload containing base64 encoded strings for the extracted images, along with metadata parsed directly from the generated `track_points.json`.

```json
{
  "foreground_base64": "iVBORw0KGgoAAAANSUhEUg...",
  "gameplay_base64": "...",
  "background_base64": "...",
  "metadata": {
    "version": "1.0",
    "timestamp": "2026-04-23T12:00:00.000Z",
    "source_image": "input.jpg",
    "aspect_ratio": 1.7778,
    "track_color": "#9e5752",
    "points": [ [0.1, 0.5], [0.2, 0.45] ]
  }
}
```

*Note: The `metadata` field currently maps directly to the contents of `track_points.json` (as seen in `samples/output/track_points.json` and `samples/output/example.json`).*

## Intended Future LevelPayload

The following is the intended future payload format that the Windows Server and Unity client will expect.

```json
{
  "status": "ok",
  "images": {
    "foreground_base64": "...",
    "gameplay_base64": "...",
    "background_base64": "..."
  },
  "track": {
    "points": [
      {"x": 0.1, "y": 0.8}
    ],
    "terrain_type": "road",
    "friction": 0.8
  },
  "metadata": {
    "source_width": 1024,
    "source_height": 768
  }
}
```

### Mismatches

⚠️ **There are significant differences between the current and intended format:**

1.  **Structure:** The intended format wraps images in an `images` object and adds a `status` field.
2.  **Track Points:** The current format provides points as arrays `[x, y]`. The intended format uses objects `{"x": ..., "y": ...}` and groups them under a `track` object.
3.  **Track Properties:** The intended format includes `terrain_type` and `friction`, which are currently not generated. The current format has `track_color`.
4.  **Metadata:** The intended format expects `source_width` and `source_height`. The current format provides `aspect_ratio`, `timestamp`, `version`, and `source_image`.
