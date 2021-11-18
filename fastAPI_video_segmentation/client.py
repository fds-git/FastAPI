import requests
import os
from PIL import Image
import io
from websocket import create_connection
import time
import cv2
import numpy as np
import websocket
import matplotlib.pyplot as plt
from typing import Tuple


def video_processing(socket: str, fps: int, width: int, height: int, mode: str = 'websocket', blur_value: Tuple[int] = None, background_path: str = None):
    '''Функция обработки видео с вебкамеры путем отправки каждого кадра на сервер "socket"
    Входные параметры:
    socket: str - устройство, на котором будет выполняться модель, например "127.0.0.1:12000"
    fps: int - количество кадров, захватываемых с веб камеры в секунду
    width: int - ширина захватываемого изображения в пикселах
    height: int - высота захватываемого изображения в пиксалах
    mode: str - способ обмена данными с сервером, возможные значения: "websocket", "post"
    blur_value: Tuple[int] - коэффициенты размытия, например (51, 51)
    background_path: str - путь до изображения, которое будет использоваться как фон'''

    try:
        response = requests.get('http://' + socket + '/')
        print(response.text)
    except:
        raise ConnectionError(f'Resource {socket} is not available')


    if background_path:
        try:
            if not os.path.exists(background_path):
                raise FileNotFoundError(f'File {background_path} is not exist')
            else:
                background_image = cv2.imread(background_path)
                background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
                background_image = cv2.resize(background_image, (width, height), interpolation = cv2.INTER_AREA)
        except Exception as e:
            background_path = None
            print(e)
            print("Background image error. Background will not be use in this session")

    if mode != 'websocket' and mode != 'post':
        raise ValueError('"Mode" value must be equal "websocket" or "post"')

    video_session = cv2.VideoCapture(0)
    video_session.set(cv2.CAP_PROP_FPS, fps)
    video_session.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video_session.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if mode == 'websocket':
        web_socket = create_connection('ws://' + socket + '/ws')

    while True:
        ret, numpy_image = video_session.read()
        numpy_image_rgb = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

        buf = io.BytesIO()
        np.save(buf, numpy_image_rgb, allow_pickle=True)
        bytes_image = buf.getvalue()

        if mode == 'websocket':
            web_socket.send_binary(bytes_image)
            response =  web_socket.recv()
            buf = io.BytesIO(response)
        else:   # mode == 'post'
            response = requests.post('http://' + socket + '/predict', data=bytes_image)
            buf = io.BytesIO(response.content)

        numpy_mask = np.load(buf, allow_pickle=True)

        if blur_value and not background_path:
            is_background = np.sum(numpy_mask, axis=2)
            is_background = is_background == 0
            blur = cv2.GaussianBlur(numpy_image, blur_value, 0)
            numpy_image[is_background] = blur[is_background]

        if background_path:
            is_background = np.sum(numpy_mask, axis=2)
            is_background = is_background == 0
            numpy_image[is_background] = background_image[is_background]

        cv2.imshow("image", numpy_image)
        cv2.imshow("mask",  numpy_mask)
        if cv2.waitKey(10) == 27: # Клавиша Esc
            break

    video_session.release()
    cv2.destroyAllWindows()

    if mode == 'websocket':
        web_socket.close()



params = {
'socket':'127.0.0.1:12000', 
'fps':10, 
'width':640, 
'height':480, 
'mode':'websocket', 
'background_path':'./background.jpg', 
'blur_value':(51, 51)
}

video_processing(**params)