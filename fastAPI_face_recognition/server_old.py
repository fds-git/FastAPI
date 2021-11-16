from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import io
import torch
import uvicorn
import numpy as np
import cv2
import pandas as pd
import json

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


PROB_TREASHOLD = 0.9
DISTANCE_TREASHOLD = 1.0 # Пороговое расстояние между эмбэддингами лиц

app = FastAPI()
conn_mgr = ConnectionManager()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(keep_all=True, device='cuda:0')        
recognizer = InceptionResnetV1(pretrained='vggface2', device=device).eval()
#my_model = NeuralNetwork(device=device)
# pickle сохраняет типы данных в колонках (в данном случае centroids - ndarray-объекты)
database = pd.read_pickle('database.pkl')
# Получаем из колонки 'centroid' датафрейма двумерный массив, содержащий координаты всех центроид,
# а затем преобразуем его в двумерный тензор для более быстрой обработки
centroids = np.vstack(database['centroid'].values)
centroids = torch.from_numpy(centroids).to(device)


@app.get("/")
async def post_endpoint():
    return {'message': 'carvana mask prediction model'}


# Предсказание через post запрос
@app.post("/predict")
async def post_endpoint(request: Request):
    bytes_frame = await request.body()  # async read
    buf = io.BytesIO(bytes_frame)
    frame = np.load(buf, allow_pickle=True)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs = detector.detect(rgb_frame, landmarks=False)
    starting_points = []
    ending_points = []
    names = []
    if boxes is not None:
        for box, prob in zip(boxes, probs):
            if prob >= PROB_TREASHOLD:
                try:
                    left, top, right, bottom = (int(box[0])), (int(box[1])), (int(box[2])), (int(box[3]))
                    starting_point = (left, top)
                    ending_point = (right, bottom)

                    face_numpy = rgb_frame[top:bottom, left:right, :]
                    face_numpy = cv2.resize(face_numpy, (160, 160)).astype('float')
                    #face_numpy = face_numpy/255.0
                    face_tensor = torch.from_numpy(face_numpy).to(device)
                    face_tensor = face_tensor.permute(2,0,1)
                    face_tensor = torch.unsqueeze(face_tensor, 0)
                    face_tensor = face_tensor.float()
                    # Встроенная функция стандартизации приводит входные данные к виду, подходящему
                    # для обработки нейронной сетью
                    face_tensor = fixed_image_standardization(face_tensor)
                    face_embedding = recognizer(face_tensor).detach()

                    centroid_distances = torch.linalg.vector_norm((centroids - face_embedding), ord=2, dim=1)
                    print(centroid_distances)
                    idx_min = torch.argmin(centroid_distances).item()
                    distance_min = torch.min(centroid_distances).item()
                    if distance_min >= DISTANCE_TREASHOLD:
                        nearest_name = 'unknown'
                    else:
                        nearest_name = database['name'][idx_min]

                    starting_points.append(starting_point)
                    ending_points.append(ending_point)
                    names.append(nearest_name)

                except:
                    continue

    result = {'starting_points': starting_points, 'ending_points': ending_points, 'names': names}
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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            starting_points = []
            ending_points = []
            names = []
            boxes, probs = detector.detect(rgb_frame, landmarks=False)
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob >= PROB_TREASHOLD:
                        try:
                            left, top, right, bottom = (int(box[0])), (int(box[1])), (int(box[2])), (int(box[3]))
                            starting_point = (left, top)
                            ending_point = (right, bottom)

                            face_numpy = rgb_frame[top:bottom, left:right, :]
                            face_numpy = cv2.resize(face_numpy, (160, 160)).astype('float')
                            #face_numpy = face_numpy/255.0
                            face_tensor = torch.from_numpy(face_numpy).to(device)
                            face_tensor = face_tensor.permute(2,0,1)
                            face_tensor = torch.unsqueeze(face_tensor, 0)
                            face_tensor = face_tensor.float()
                            # Встроенная функция стандартизации приводит входные данные к виду, подходящему
                            # для обработки нейронной сетью
                            face_tensor = fixed_image_standardization(face_tensor)
                            face_embedding = recognizer(face_tensor).detach()

                            centroid_distances = torch.linalg.vector_norm((centroids - face_embedding), ord=2, dim=1)
                            print(centroid_distances)
                            idx_min = torch.argmin(centroid_distances).item()
                            distance_min = torch.min(centroid_distances).item()
                            if distance_min >= DISTANCE_TREASHOLD:
                                nearest_name = 'unknown'
                            else:
                                nearest_name = database['name'][idx_min]

                            starting_points.append(starting_point)
                            ending_points.append(ending_point)
                            names.append(nearest_name)
                        except:
                            continue

            result = {'starting_points': starting_points, 'ending_points': ending_points, 'names': names}
            bytes_result = json.dumps(result).encode('utf-8')
            await websocket.send_bytes(bytes_result)

    except WebSocketDisconnect as e: # Клиент сам отключился
        conn_mgr.disconnect(websocket)
    except:
        websocket.close()
        conn_mgr.disconnect(websocket)