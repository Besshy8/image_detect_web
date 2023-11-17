from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import asyncio

from ultralytics import YOLO

app = FastAPI()

# HTMLページのテンプレート
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Video Streaming</title>
    </head>
    <body>
        <h1>Video Streaming with FastAPI and OpenCV</h1>
        <img id="video" src="">
        <script>
            var img = document.getElementById('video');
            var ws = new WebSocket('ws://localhost:8080/ws');
            ws.onmessage = function(event) {
                // BlobオブジェクトのURLを生成して、イメージのソースに設定
                var blob = event.data;
                if (blob instanceof Blob) {
                    img.src = URL.createObjectURL(blob);
                }
            };
        </script>
    </body>
</html>
"""



@app.get("/")
async def get():
    # HTMLレスポンスを返す
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # カメラのキャプチャ開始
    try:
        while True:
            ret, frame = cap.read()  # フレームを読み込む
            ##print(ret)
            if not ret:
                break
            model = YOLO("../../yolo/weight/last.pt")
            results = model(frame)
            img_annotated = results[0].plot()
            _, buffer = cv2.imencode('.jpg', img_annotated)  # JPEG形式にエンコード
            #_, buffer = cv2.imencode('.jpg', frame)  # JPEG形式にエンコード
            frame = buffer.tobytes()
            await websocket.send_bytes(frame)  # WebSocket経由で送信
            await asyncio.sleep(0.1)  # フレームレートを制御
    except WebSocketDisconnect:
        cap.release()  # WebSocketが切断されたらリソースを解放



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8080)




