import torch.nn as nn
import torch
import numpy as np


class NeuralNetwork(nn.Module):
    '''Класс для работы с нейронной сетью YOLO V5'''

    def __init__(self, device: str):
        '''Конструктор класса
        Входные параметры:
        device: str - устройство, на котором будет выполняться модель
        Возвращаемые значения: 
        объект класса NeuralNetwork'''

        super(NeuralNetwork, self).__init__()

        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s').eval().to(self.device)

    
    def predict(self, numpy_image: np.ndarray) -> object:
        '''Метод предсказания нейронной сетью классов и bbox'ов
        Входные параметры:
        numpy_image: np.ndarray - входное изображение
        Возвращаемые значения:
        results: object - результат работы модели'''

        self.model.eval()
        results = self.model(numpy_image)#, size=640)
        return results