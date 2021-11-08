import requests
import glob
import os
from pydantic import BaseModel
from typing import List
from PIL import Image
import io
source_directory = './client_src_images/'
destination_directory = './client_dst_images/'

# Адрес сервера и порт
url = "http://127.0.0.1:12000"

class ServerFiles(BaseModel):
    file_names: List[str]

    
#-------------------------------------------------------

def save_file(source_directory: str, url: str):
    '''Функция POST отправляет все файлы из директории source_directory на сервер url,
    для сохранения на сервере
    Входные параметры:
    source_directory: str - директория с исходными файлами клиента
    url: str - адрес, порт и метод сервера'''
    
    file_list = os.listdir(source_directory)
    for file_name in file_list:
        files = {'in_file': open(source_directory + file_name, 'rb')}
        response = requests.post(url, files=files)
        print(response)


def load_file(destination_directory: str, file_names: list, url: str):
    '''Функция POST скачивает файлы file_names с сервера url и сохраняет в destination_directory,
    Входные параметры:
    destination_directory: str - директория для сохранения файлов на клиенте
    file_names: list - имена файлов на сервере
    url: str - адрес, порт и метод сервера'''
    
    for file_name in file_names:
        response = requests.post(url +'/'+ file_name)
        print(response)
        file_content = response.content
        with open(destination_directory + file_name, 'wb') as binary_file:
            binary_file.write(file_content)


def change_file_save(source_directory: str, destination_directory: str, url: str):
    '''Функция POST отправляет файлы из директории source_directory на сервер url,
    сохраняет их на сервере, принимает измененные файлы возвращает и сохраняет их в destination_directory
    Входные параметры:
    source_directory: str - директория с исходными файлами клиента
    destination_directory: str - директория клиента для сохранения измененных сервером файлов
    url: str - адрес, порт и метод сервера'''
    
    file_list = os.listdir(source_directory)
    for file_name in file_list:
        files = {'in_file': open(source_directory + file_name, 'rb')}
        response = requests.post(url, files=files)
        print(response)
        # Допускаем, что сервер мог поменять имена файлов
        file_name = response.headers['content-disposition'].split('"')[1]
        file_content = response.content
        with open(destination_directory + file_name, 'wb') as binary_file:
            binary_file.write(file_content)


def change_file(source_directory: str, destination_directory: str, url: str):
    '''Функция POST отправляет файлы из директории source_directory на сервер url без сохранения на сервере,
    принимает измененые файлы и сохраняет их в destination_directory
    Входные параметры:
    source_directory: str - директория с исходными файлами клиента
    destination_directory: str - директория клиента для сохранения измененных сервером файлов
    url: str - адрес, порт и метод сервера'''

    file_list = os.listdir(source_directory)
    for file_name in file_list:
        files = {'in_file': open(source_directory + file_name, 'rb')}
        response = requests.post(url, files=files)
        #------------------- для проверки--------------------------
        #pil_image = Image.open(io.BytesIO(response.content))
        #pil_image.show()
        #------------------- для проверки--------------------------
        with open(destination_directory + file_name, 'wb') as binary_file:
            binary_file.write(response.content)


def get_file_names(url: str) -> List[str]:
    '''Функция POST запрашивает имена текстовых файлов, хранящихся на сервере и возвращает их
    в виде списка строк 
    Входные параметры:
    url: str - адрес, порт и метод сервера
    Выходные значения:
    file_names: List[str] - список с именами файлов, хранящихся на сервере'''
    
    response = requests.post(url)
    file_names = response.json()['file_names']
    return file_names



#save_file(source_directory, url + '/save_file')
#file_names = get_file_names(url + '/get_file_names')
#load_file(destination_directory, file_names, url + '/load_file')

#change_file_save(source_directory, destination_directory, url + '/change_file_save')
change_file(source_directory, destination_directory, url + '/change_file')


#print(file_names)
