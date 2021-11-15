from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
import io
import torch
from model import NeuralNetwork
import numpy as np

app = FastAPI()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
my_model = NeuralNetwork(device=device)


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

conn_mgr = ConnectionManager()

@app.get("/")
async def post_endpoint():
    return {'message': 'carvana mask prediction model'}


# Предсказание через post запрос
@app.post("/predict")
async def post_endpoint(request: Request):
    bytes_image = await request.body()  # async read
    buf = io.BytesIO(bytes_image)
    numpy_image = np.load(buf, allow_pickle=True)

    results = my_model.predict(numpy_image)
    # Накладываем bboxes на изображение
    results.render()

    buf = io.BytesIO()
    np.save(buf, results.imgs[0], allow_pickle=True)
    bytes_mask = buf.getvalue()
    
    return Response(content=bytes_mask)


# Предсказание через websocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await conn_mgr.connect(websocket)
    
    try:
        while True:
            bytes_image = await websocket.receive_bytes()
            buf = io.BytesIO(bytes_image)
            numpy_image = np.load(buf, allow_pickle=True)

            results = my_model.predict(numpy_image)
            # Накладываем bboxes на изображение
            results.render()

            buf = io.BytesIO()
            np.save(buf, results.imgs[0], allow_pickle=True)
            bytes_mask = buf.getvalue()

            await websocket.send_bytes(bytes_mask)
    except WebSocketDisconnect as e: # Клиент сам отключился
        conn_mgr.disconnect(websocket)
    except:
        websocket.close()
        conn_mgr.disconnect(websocket)