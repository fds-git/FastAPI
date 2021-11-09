from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image
from starlette.responses import StreamingResponse
import io
import torch
from model import NeuralNetwork
import os
import uvicorn
import PIL.Image


app = FastAPI()
path_to_model = './model_lab_v3.pth'
if not os.path.isfile(path_to_model):
    raise FileNotFoundError('Model not found')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
my_model = NeuralNetwork(device=device, path_to_model=path_to_model)

MASK_TREASHOLD = 0.5
OUTPUT_SIZE = (1280, 1918)


class ConnectionManager:
    """Web socket connection manager."""

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
async def post_endpoint(in_file: UploadFile=File(...)):
    image_bytes = await in_file.read()  # async read
    pil_image = Image.open(io.BytesIO(image_bytes))
    tensor_image = my_model.preprocessing(pil_image)
    tensor_mask = my_model.predict(tensor_image)
    pil_mask = NeuralNetwork.postprocessing(tensor_mask, output_size=OUTPUT_SIZE, mask_treashold=MASK_TREASHOLD)

    # Обязательно так делать для PIL иначе у клиента ошибка будет
    buf = io.BytesIO()
    pil_mask.save(buf, "PNG")
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")


# Предсказание через websocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    #await websocket.accept()
    await conn_mgr.connect(websocket)
    
    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            pil_image = Image.open(io.BytesIO(image_bytes))
            tensor_image = my_model.preprocessing(pil_image)
            tensor_mask = my_model.predict(tensor_image)
            pil_mask = NeuralNetwork.postprocessing(tensor_mask, output_size=OUTPUT_SIZE, mask_treashold=MASK_TREASHOLD)

            # Обязательно так делать для PIL иначе у клиента ошибка будет
            buf = io.BytesIO()
            pil_mask.save(buf, "PNG")
            buf.seek(0)
            await websocket.send_bytes(buf)
    except WebSocketDisconnect as e: # Клиент сам отключился
        conn_mgr.disconnect(websocket)
    except:
        websocket.close()
        conn_mgr.disconnect(websocket)



#if __name__ == '__main__':
#    uvicorn.run(app=app, host="0.0.0.0", port=12000, workers=4)