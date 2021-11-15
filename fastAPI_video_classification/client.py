import requests
import io
from websocket import create_connection
import cv2
import numpy as np
import websocket


def video_processing(socket: str, fps: int, width: int, height: int, mode: str = 'websocket'):
    '''Функция обработки видео с вебкамеры путем отправки каждого кадра на сервер "socket"
    для решения задачи object detection с помощью YOLO V5. Сервер возвращает изображение,
    с нанесенными на нем bbox'ами и классами
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
        ret, numpy_input_image = video_session.read()

        buf = io.BytesIO()
        np.save(buf, numpy_input_image, allow_pickle=True)
        bytes_input_image = buf.getvalue()

        if mode == 'websocket':
            web_socket.send_binary(bytes_input_image)
            response =  web_socket.recv()
            buf = io.BytesIO(response)
        else:   # mode == 'post'
            response = requests.post('http://' + socket + '/predict', data=bytes_input_image)
            buf = io.BytesIO(response.content)

        numpy_output_image = np.load(buf, allow_pickle=True)


        cv2.imshow("YOLO V5", numpy_output_image)
        if cv2.waitKey(10) == 27: # Клавиша Esc
            break

    video_session.release()
    cv2.destroyAllWindows()

    if mode == 'websocket':
        web_socket.close()



params = {
'socket':'127.0.0.1:12000', 
'fps':1, 
'width':1280, 
'height':720, 
'mode':'websocket', 
}

video_processing(**params)