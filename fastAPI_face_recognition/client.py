import requests
import io
from websocket import create_connection
import cv2
import numpy as np
import websocket
import json

FONT_SCALE = 1.0
TEXT_THICKNESS = 1
LINE_THICKNESS = 2
BLUE_COLOR = (255, 0, 0)
WHITE_COLOR = (255, 255, 255)


def video_processing(socket: str, fps: int, width: int, height: int, mode: str = 'websocket'):
    '''Функция обработки видео с вебкамеры путем отправки каждого кадра на сервер "socket"
    Входные параметры:
    socket: str - устройство, на котором будет выполняться модель, например "127.0.0.1:12000"
    fps: int - количество кадров, захватываемых с веб камеры в секунду
    width: int - ширина захватываемого изображения в пикселах
    height: int - высота захватываемого изображения в пиксалах
    mode: str - способ обмена данными с сервером, возможные значения: "websocket", "post"'''

    try:
        response = requests.get('http://' + socket + '/')
        print(response.text)
    except:
        raise ConnectionError(f'Resource {socket} is not available')


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

        buf = io.BytesIO()
        np.save(buf, numpy_image, allow_pickle=True)
        bytes_image = buf.getvalue()

        if mode == 'websocket':
            web_socket.send_binary(bytes_image)
            response =  web_socket.recv()
            response = json.loads(response)
        else:   # mode == 'post'
            response = requests.post('http://' + socket + '/predict', data=bytes_image)
            response = response.json()

        for i in range(len(response['names'])):
            cv2.rectangle(numpy_image, response['starting_points'][i], response['ending_points'][i], BLUE_COLOR, LINE_THICKNESS)
            cv2.putText(numpy_image, response['names'][i], (response['starting_points'][i][0] + 6, response['ending_points'][i][1]  - 6), 
                cv2.FONT_HERSHEY_DUPLEX, FONT_SCALE, WHITE_COLOR, TEXT_THICKNESS)
        cv2.imshow('Video', numpy_image)

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
}

video_processing(**params)