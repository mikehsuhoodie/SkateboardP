# Monorepo Integration

## Recommended Future Monorepo Structure

```text
skatep/
├── unity-client/
├── server-win/
├── wsl-cv/
├── shared/
└── docs/
```

## Module Responsibilities

*   **`unity-client/`**: Contains the Unity game project. All Unity C# scripts belong in `unity-client/Assets/Scripts/`.
*   **`server-win/`**: The Windows server application. All Windows server Python code belongs here.
*   **`wsl-cv/`**: The computer vision backend. All WSL CV Python code (this repository) belongs here.
*   **`shared/`**: Shared API schemas, protobufs, and sample payloads.
*   **`docs/`**: Global project documentation.

## Important Rule

**Do not place WSL CV or Windows server files under Unity `Assets/Scripts/`.** They must remain in their respective root folders to ensure clean dependency management and separation of concerns.
