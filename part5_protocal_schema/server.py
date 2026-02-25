from flask import Flask, request, send_file
import os

app = Flask(__name__)

# 確保目錄存在
if not os.path.exists('captures'):
    os.makedirs('captures')

# 儲存圖片的路徑
SAVE_PATH = os.path.join('captures', 'latest.jpg')

@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' in request.files:
        photo = request.files['photo']
        # 一律存成 latest.jpg，方便下載測試
        photo.save(SAVE_PATH)
        print(f"成功接收並儲存: {SAVE_PATH}")
        return "OK", 200
    return "No photo", 400

@app.route('/download', methods=['GET'])
def download():
    # 檢查檔案是否存在
    if os.path.exists(SAVE_PATH):
        print(f"正在傳送圖片給 Unity...")
        return send_file(SAVE_PATH, mimetype='image/jpeg')
    else:
        print(f"錯誤：找不到檔案 {SAVE_PATH}")
        return "File not found", 404

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)