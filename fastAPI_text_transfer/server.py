from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import aiofiles
from pydantic import BaseModel
import os

app = FastAPI()
out_file_path = './server_files/'


# Сохраняет текстовый файл на сервер
@app.post("/save")
async def post_endpoint(in_file: UploadFile=File(...)):
    async with aiofiles.open(out_file_path + in_file.filename, 'wb') as out_file:
        content = await in_file.read()  # async read
        await out_file.write(content)  # async write
    return {"Result": "OK"}


# Передает текстовый файл клиенту
@app.post("/load/{file_name}")
async def post_endpoint(file_name: str):
    return FileResponse(path=out_file_path + file_name, filename=file_name, media_type='text')


# Сохраняет текстовые файлы на сервер и возвращает обратно клиенту
@app.post("/save_get_back")
async def post_endpoint(in_file: UploadFile=File(...)):
    async with aiofiles.open(out_file_path + in_file.filename, 'wb') as out_file:
        content = await in_file.read()  # async read
        await out_file.write(content)  # async write
    return FileResponse(path=out_file_path + in_file.filename, filename=out_file_path + in_file.filename, media_type='text')
    #return {"Result": "OK"}


# Изменяет текстовый файл и возвращает его без сохранения на сервере
@app.post("/change_get_back")
async def post_endpoint(in_file: UploadFile=File(...)):
    content = await in_file.read()  # async read
    print(content)  
    content_str = content.decode("utf-8")
    print(content_str)
    content_str = content_str + " 2020202020"
    print(content_str)
    #content = content_str.encode("utf-8")
    
    print(content)
    return content_str
    #print(content)


# Возвращает строку с именами всех текстовых файлов на сервере
@app.post("/list_text_files")
async def post_endpoint(out_file_path=out_file_path):
    file_list = os.listdir(out_file_path)
    return file_list
