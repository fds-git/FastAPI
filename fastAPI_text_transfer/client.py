import requests
import glob
import os
from pydantic import BaseModel
source_directory = './client_src_files/'
destination_directory = './client_dst_files/'

# Адрес сервера и порта
url = "http://127.0.0.1:12000"

# Сохранение файлов на сервере
url_save_file = "http://127.0.0.1:12000/save"
# Загрузка файлов из сервера
url_load_file = "http://127.0.0.1:12000/load"
# Сохранение файлов не сервере и возвращение клиенту
url_save_file_get_back = "http://127.0.0.1:12000/save_get_back"
# Изменение файлов не сервере и возврат клиенту без сохранения на сервере
url_change_file_get_back = "http://127.0.0.1:12000/change_get_back"
# Запрос списка имен текстовых файлов на сервере
url_get_list_text_files = "http://127.0.0.1:12000/list_text_files"


files = ['er.txt', 'qwe.txt']


    # для картинок попробовать
    #r = requests.get('https://example.com/image.jpg')
    #i = Image.open(StringIO(r.content))
    
#-------------------------------------------------------

def save(source_directory: str, url: str):
    '''Функция POST отправляет текстовые файлы из директории source_directory на сервер url,
    для сохранения на сервере
    Входные параметры:
    source_directory: str - директория с исходными файлами клиента
    url: str - адрес, порт и метод сервера'''
    
    file_list = os.listdir(source_directory)
    for file_name in file_list:
        files = {'in_file': open(source_directory + file_name, 'rb')}
        response = requests.post(url, files=files)
        print(response)


def load(destination_directory: str, file_names: list, url: str):
    '''Функция POST скачивает файлы file_names с сервера url и сохраняет в destination_directory,
    Входные параметры:
    destination_directory: str - директория для сохранения файлов на клиенте
    file_names: list - имена файлов на сервере
    url: str - адрес, порт и метод сервера'''
    
    for file_name in file_names:
        response = requests.post(url +'/'+ file_name)
        file_content = response.content
        with open(destination_directory + file_name, 'wb') as binary_file:
            binary_file.write(file_content)


def save_get_back(source_directory: str, destination_directory: str, url: str):
    '''Функция POST отправляет текстовые файлы из директории source_directory на сервер url,
    сохраняет их на сервере, возвращает и сохраняет их в destination_directory
    Входные параметры:
    source_directory: str - директория с исходными файлами клиента
    destination_directory: str - директория клиента для сохранения измененных сервером файлов
    url: str - адрес, порт и метод сервера'''
    
    file_list = os.listdir(source_directory)
    for file_name in file_list:
        files = {'in_file': open(source_directory + file_name, 'rb')}
        response = requests.post(url, files=files)
        # Допускаем, что сервер мог поменять имена файлов
        file_name = response.headers['content-disposition'].split('/')[-1][:-1]
        file_content = response.content
        with open(destination_directory + file_name, 'wb') as binary_file:
            binary_file.write(file_content)


def change_get_back(source_directory: str, destination_directory: str, url: str):
    '''Функция POST отправляет текстовые файлы из директории source_directory на сервер url,
    принимает измененые файлы и сохраняет их в destination_directory
    Входные параметры:
    source_directory: str - директория с исходными файлами клиента
    destination_directory: str - директория клиента для сохранения измененных сервером файлов
    url: str - адрес, порт и метод сервера'''

    file_list = os.listdir(source_directory)
    for file_name in file_list:
        files = {'in_file': open(source_directory + file_name, 'rb')}
        response = requests.post(url, files=files)
        file_content = response.content
        with open(destination_directory + file_name, 'wb') as binary_file:
            binary_file.write(file_content)


def list_text_files(url: str) -> str:
    '''Функция POST запрашивает имена текстовых файлов, хранящихся на сервере и возвращает их
    в виде списка строк 
    Входные параметры:
    url: str - адрес, порт и метод сервера
    Выходные значения:
    file_names: str - строка с именами файлов, хранящихся на сервере'''
    
    response = requests.post(url)
    print((response.content).decode())
    return response


#save(source_directory, url + '/save')
#load(destination_directory, files, url + '/load')
#save_get_back(source_directory, destination_directory, url + '/save_get_back')
change_get_back(source_directory, destination_directory, url + '/change_get_back')
#result = list_text_files(url + '/list_text_files')
#print(result)
