# 2.5D Skateboard Path Vision Project (Mike v2.0)

## 1. 專案願景 (The Reboot Goal)
這是一個將現實照片轉化為 2.5D 滑板遊戲關卡的系統。
## 2. 環境設定 (Environment Setup)
### 軟體需求
- **Unity**: 2022.3.x LTS (2D Core Template)
- **Python**: 3.10.12
- **關鍵套件**:
  - Python: `opencv-python`, `numpy`, `matplotlib`
  - Unity: `Newtonsoft.Json` (用於解析 Data Contract)

### 快速啟動 (The Reboot Button)
1. **複製倉庫**:
   `git clone [your-repo-url]`
2. **Python 環境初始化**:
   ```bash
   cd core_vision
   python -m venv venv
   source venv/bin/activate  # Windows 使用 venv\Scripts\activate
   pip install -r requirements.txt