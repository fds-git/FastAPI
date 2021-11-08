from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.responses import Response
import uvicorn
import aiofiles
from pydantic import BaseModel
import os
from PIL import Image
from starlette.responses import StreamingResponse
import io
from typing import List
import matplotlib.pyplot as plt


class ServerFiles(BaseModel):
    file_names: List[str]


app = FastAPI()
out_file_path = './server_images/'


# Сохраняет файл на сервер
@app.post("/save_file")
async def post_endpoint(in_file: UploadFile=File(...)):
    async with aiofiles.open(out_file_path + in_file.filename, 'wb') as out_file:
        content = await in_file.read()  # async read
        await out_file.write(content)  # async write
    #return {"Result": "OK"}


# Передает файл клиенту
@app.post("/load_file/{file_name}")
async def post_endpoint(file_name: str):
    if os.path.exists(out_file_path + file_name):
        return FileResponse(path=out_file_path + file_name, filename=file_name, media_type="image/jpg")
    return {"Error": "File not found"}


# Сохраняет файлы на сервер, изменяет их разрешение и возвращает обратно клиенту
@app.post("/change_file_save")
async def post_endpoint(in_file: UploadFile=File(...)):
    async with aiofiles.open(out_file_path + in_file.filename, 'wb') as out_file:
        content = await in_file.read()  # async read
        await out_file.write(content)  # async write
        image = Image.open(out_file_path + in_file.filename)
        image = image.resize((512, 512))
        image.save(out_file_path + in_file.filename)
    if os.path.exists(out_file_path + in_file.filename):
    	return FileResponse(path=out_file_path + in_file.filename, filename=in_file.filename, media_type="image/jpg")
    return {"Error": "File not found"}


def read_image(image_encoded):
    pil_image = Image.open(io.BytesIO(image_encoded))
    return pil_image


# Изменяет файл и возвращает его без сохранения на сервере
@app.post("/change_file")
async def post_endpoint(in_file: UploadFile=File(...)):
    content = await in_file.read()  # async read
    #print((content))
    image = read_image(content)
    image = image.resize((512,512))
    
    #image.show() # - для проверки
    
    # C PIL надо обязательно так делать, иначе ошибка будет на клиенте
    buf = io.BytesIO()
    image.save(buf, "JPEG")
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/jpg")

    


# Передает клиенту объект ServerFiles с именами всех файлов в директории out_file_path сервера
@app.post("/get_file_names", response_model=ServerFiles)
async def post_endpoint(out_file_path=out_file_path):
    file_names = os.listdir(out_file_path)
    return ServerFiles(file_names=file_names)

