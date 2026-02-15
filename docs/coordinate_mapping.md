# Coordinate Mapping Specification

To convert the 2D path extracted from a photo into a Unity 2D environment, we use a normalized-to-world mapping logic.

## 1. Input Space (Image)
The Computer Vision (CV) layer outputs points in a **Normalized Space**:
- **(0.0, 0.0)**: Bottom-Left of the image.
- **(1.0, 1.0)**: Top-Right of the image.

This ensures the path remains relative to the aspect ratio, regardless of the physical resolution of the photo.

## 2. Target Space (Unity 2D)
In Unity, the orthographic camera defines the visible world:
- `Camera.main.orthographicSize`: Half the height of the camera view in world units.
- `Camera.main.aspect`: The ratio of width to height.

## 3. Conversion Formula
Given a normalized point $(x_n, y_n)$:

$$Unity_y = (y_n - 0.5) \times (\text{OrthoSize} \times 2)$$
$$Unity_x = (x_n - 0.5) \times (\text{OrthoSize} \times 2 \times \text{Aspect})$$

### Example
If `orthographicSize = 5.0` (Height = 10 units) and `aspect = 1.77` (Width = 17.7 units):
- A point at $(0.5, 0.5)$ becomes $(0, 0)$ in Unity.
- A point at $(0.0, 0.0)$ becomes $(-8.85, -5.0)$ in Unity.

## 4. Implementation in C#
```csharp
Vector2 ConvertToUnity(float xn, float yn, float orthoSize, float aspect) {
    float h = orthoSize * 2.0f;
    float w = h * aspect;
    float ux = (xn - 0.5f) * w;
    float uy = (yn - 0.5f) * h;
    return new Vector2(ux, uy);
}
```
