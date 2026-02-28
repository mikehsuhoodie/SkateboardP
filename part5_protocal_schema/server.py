from flask import Flask, request, send_file, jsonify
import os
import base64, json

app = Flask(__name__)

# 確保目錄存在
if not os.path.exists('captures'):
    os.makedirs('captures')

# 儲存圖片的路徑
SAVE_PATH = os.path.join('captures', 'latest.jpg')
SCHEMA_PATH = os.path.join('captures', 'test_smoothed.json')
# 簡易驗證金鑰 (在學校網路建議加上)
API_KEY = "112550191"

def check_auth():
    return request.headers.get("X-API-KEY") == API_KEY

@app.route('/upload', methods=['POST'])
def upload():
    if not check_auth():
        return "Unauthorized", 401
    print(f"成功連線")
    if 'photo' in request.files:
        photo = request.files['photo']
        # 一律存成 latest.jpg，方便下載測試
        photo.save(SAVE_PATH)
        print(f"成功接收並儲存: {SAVE_PATH}")
        return "OK", 200
    return "No photo", 400

@app.route('/download', methods=['GET'])
def download():
    if not check_auth():
        return "Unauthorized", 401
    # 檢查檔案是否存在
    if os.path.exists(SAVE_PATH) and os.path.exists(SCHEMA_PATH):
        with open(SAVE_PATH, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        with open(SCHEMA_PATH,"r") as json_file:
            schema_file = json.load(json_file)
        print(f"正在傳送圖片給 Unity...")
        return jsonify({
                "status": "success",
                "image_base64": encoded_string,
                "metadata": schema_file
            })

        # {
        # "image_base64": "...",
        # "image_name": "result.jpg",
        # "status_code": 200
        # }

    else:
        print(f"錯誤：找不到檔案 {SAVE_PATH} 或 {SCHEMA_PATH}")
        return "File not found", 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)