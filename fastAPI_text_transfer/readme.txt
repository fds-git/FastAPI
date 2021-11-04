Первое клиент-серверное приложение для изучения FastAPI
представлены функции передачи через post-запросы текстовых файлов от клиента к серверу 
и наоборот с их сохранением на сервере

install python-multipart
uvicorn server:app --host 0.0.0.0 --port 12000 --reload
server - server.py
app - FasAPI application name in server.py
