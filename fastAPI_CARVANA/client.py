import requests
import os
from PIL import Image
import io
from websocket import create_connection
import time

source_directory = './client_src_images/'
destination_directory = './client_dst_images/'
url = "http://127.0.0.1:12000"
websocket_url = "ws://127.0.0.1:12000/ws"

def process_img_post(source_directory: str, destination_directory: str, url: str):
    '''Функция для отправки изображений из source_directory на сервер url через post запрос,
    получения обработанных изображений и их сохранения в destination_directory
    Входные параметры:
    source_directory: str - директория с исходными изображениями
    destination_directory: str - директория для сохранения предсказанных масок
    url: str - адрес сервера'''

    file_list = os.listdir(source_directory)
    for file_name in file_list:
        if file_name.split('.')[-1] == 'jpg':
            files = {'in_file': open(source_directory + file_name, 'rb')}
            response = requests.post(url, files=files)
            print(response)
            #------------------- для проверки--------------------------
            #pil_image = Image.open(io.BytesIO(response.content))
            #pil_image.show()
            #print(file_name)
            #------------------- для проверки--------------------------
            with open(destination_directory + (file_name.split('.')[0] + '.png'), 'wb') as binary_file:
                binary_file.write(response.content)


def process_img_socket(source_directory: str, destination_directory: str, url: str):
    '''Функция для отправки изображений из source_directory на сервер url через websocket,
    получения обработанных изображений и их сохранения в destination_directory
    Входные параметры:
    source_directory: str - директория с исходными изображениями
    destination_directory: str - директория для сохранения предсказанных масок
    url: str - адрес сервера'''

    web_socket = create_connection(url)
    file_list = os.listdir(source_directory)
    for file_name in file_list:
        if file_name.split('.')[-1] == 'jpg':
            with open(source_directory + file_name, "rb") as image_file:
                web_socket.send_binary(image_file.read())
            response =  web_socket.recv()
            #------------------- для проверки--------------------------
            #img = Image.open(io.BytesIO(response))
            #img.show()
            #------------------- для проверки--------------------------
            with open(destination_directory + (file_name.split('.')[0] + '.png'), 'wb') as binary_file:
                binary_file.write(response)

    web_socket.close()



if not os.path.exists(source_directory):
    raise FileNotFoundError('Image source directory not exist')

if not os.path.exists(destination_directory):
    raise FileNotFoundError('Image destination directory not exist')

try:
    response = requests.get(url)
    print(response.text)
except:
    raise ConnectionError(f'{url} not available')



time1 = time.time()
process_img_socket(source_directory, destination_directory, websocket_url)
#process_img_post(source_directory, destination_directory, url + '/predict')
time2 = time.time()
delta = time2 - time1
print(delta)

# dima
#35.7 c - process_img_socket - 246 images
#34.7 c - process_img_post - 246 images