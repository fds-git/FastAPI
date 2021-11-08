Второе тренировочное клиент-серверное приложение для изучения FastAPI - передача картинок
представлены функции передачи через post-запросы изображений от клиента к серверу 
и наоборот с их сохранением на сервере

install python-multipart
uvicorn server:app --host 0.0.0.0 --port 12000 --reload
server - server.py
app - FasAPI application name in server.py
