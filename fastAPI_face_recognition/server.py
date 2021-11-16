from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from model import MyFaceRecognizer
import io
import torch
import uvicorn
import numpy as np
import cv2
import pandas as pd
import json
import cv2

PROB_TREASHOLD = 0.9
DISTANCE_TREASHOLD = 1.0 # Пороговое расстояние между эмбэддингами лиц

class ConnectionManager:
    """Класс для управления подключениями клиентов к вебсокетам"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client {websocket.client} is connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client {websocket.client} is disconnected")


def image_process(frame: np.ndarray, face_recognizer: MyFaceRecognizer, prob_treashold: float, distance_treashold: float) -> dict:
    '''Функция'''

    starting_points = []
    ending_points = []
    names = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = face_recognizer.detector.detect(rgb_frame, landmarks=False)
    boxes, probs = face_recognizer.box_filter(boxes, probs, prob_treashold)

    if boxes != []:
        for box, prob in zip(boxes, probs):
            try:
                face_tensor = face_recognizer.preprocessing(rgb_frame, box)
                face_embedding = face_recognizer.recognizer(face_tensor).detach()
                nearest_name = face_recognizer.get_nearest_name(face_embedding, distance_treashold)

                left, top, right, bottom = (int(box[0])), (int(box[1])), (int(box[2])), (int(box[3]))
                starting_point = (left, top)
                ending_point = (right, bottom)

                starting_points.append(starting_point)
                ending_points.append(ending_point)
                names.append(nearest_name)

            except Exception as e:
                print(e)

    return {'starting_points': starting_points, 'ending_points': ending_points, 'names': names}


app = FastAPI()
conn_mgr = ConnectionManager()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
face_recognizer = MyFaceRecognizer(device=device, database_path='database.pkl')


@app.get("/")
async def post_endpoint():
    return {'message': 'carvana mask prediction model'}


# Предсказание через post запрос
@app.post("/predict")
async def post_endpoint(request: Request):
    bytes_frame = await request.body()  # async read
    buf = io.BytesIO(bytes_frame)
    frame = np.load(buf, allow_pickle=True)

    result = image_process(frame, face_recognizer, PROB_TREASHOLD, DISTANCE_TREASHOLD)

    bytes_result = json.dumps(result).encode('utf-8')
    return Response(content=bytes_result)


# Предсказание через websocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await conn_mgr.connect(websocket)
    
    try:
        while True:
            bytes_frame = await websocket.receive_bytes()
            buf = io.BytesIO(bytes_frame)
            frame = np.load(buf, allow_pickle=True)

            result = image_process(frame, face_recognizer, PROB_TREASHOLD, DISTANCE_TREASHOLD)

            bytes_result = json.dumps(result).encode('utf-8')
            await websocket.send_bytes(bytes_result)

    except WebSocketDisconnect as e: # Клиент сам отключился
        conn_mgr.disconnect(websocket)
    except:
        websocket.close()
        conn_mgr.disconnect(websocket)