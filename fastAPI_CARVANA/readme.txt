Третье клиент-серверное приложение для изучения FastAPI на основе решения задачи CARVANA -
клиент передает изображения машин в формате .jpg из папки client_src_images на сервер,
сервер с помощью нейросети model_lab_v3.pth предсказывает для этих изображений маску и 
отправляет обратно клиенту, склиент сохраняет маски в формате .png в папке model_lab_v3.pth.
Обмен между клиентом и сервером осуществляется либо через post-запрос, либо через вебсокеты

install python-multipart
uvicorn server:app --host 0.0.0.0 --port 12000 --reload
server - server.py
app - FasAPI application name in server.py