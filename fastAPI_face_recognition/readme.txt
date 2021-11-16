Клиент-серверное приложение, выполняющее задачу распознавания лиц на изображении
с вебкамеры клиента. Клиенты отправляют изображения на на сервер через POST
запрос или вебсокеты, сервер обрабатывает их и возвращает обратно. Приложение реализовано
с помощью facenet_pytorch 

install python-multipart
uvicorn server:app --host 0.0.0.0 --port 12000 --reload
server - server.py
app - FasAPI application name in server.py
