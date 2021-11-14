import torch.nn as nn
from torch.nn import functional as F
import torch
import PIL
from albumentations import Compose
import numpy as np
import cv2
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from torchvision import models


class NeuralNetwork(nn.Module):
    '''Класс для работы с нейронной сетью для семантической сегментации изображений'''

    def __init__(self, device: str):
        '''Конструктор класса
        Входные параметры:
        device: str - устройство, на котором будет выполняться модель
        Возвращаемые значения: 
        объект класса NeuralNetwork'''

        super(NeuralNetwork, self).__init__()

        self.device = device
        #self.model = models.segmentation.deeplabv3_resnet101(pretrained=1).eval().to(self.device)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True).eval().to(self.device)
        self.transformer = A.Compose([
            #A.Resize(512, 512, cv2.INTER_AREA),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
            ToTensorV2(),
            ])

    
    def predict(self, tensor_image: torch.Tensor) -> torch.Tensor:
        '''Метод предсказания маски для изображения
        Входные параметры:
        tensor_image: torch.Tensor - изображение в тензорном формате
        Возвращаемые значения:
        tensor_masks_logit: torch.Tensor - предсказанные маски в тензорном формате (количество каналов равно количеству классов)'''

        self.model.eval()    
        tensor_masks_logit = self.model(tensor_image)
        return tensor_masks_logit['out']


    def preprocessing(self, numpy_image: np.ndarray) -> torch.Tensor:
        '''Метод предобработки данных перед подачей в сеть
        Входные параметры:
        numpy_image: np.ndarray - изображение, для которого нужно предсказать маску
        Возвращаемые значения:
        tensor_image: torch.Tensor - подготовленное для подачи в сеть изображение в тензорном формате, 
        #для которого нужно предсказать маску '''

        numpy_image = numpy_image.astype('float')/255.0
        transformed = self.transformer(image=numpy_image)
        tensor_image = transformed['image']
        tensor_image = tensor_image.unsqueeze(0)
        tensor_image = tensor_image.to(self.device).float()
        return tensor_image


    @staticmethod
    def postprocessing(tensor_mask_logit: torch.Tensor, output_size: tuple=(1280, 1918)) -> np.ndarray:
        '''Статический метод постобработки данных, полученных с выхода сети
        Входные параметры:
        tensor_mask_logit: torch.Tensor - предсказанная сетью маска в тензорном формате в масштабе logit
        число каналов равно числу классов
        output_size: tuple - пространственная размерность, к которой нужно привести выходные маски
        Возвращаемые значения:
        numpy_mask_rgb: np.ndarray - выходная маска изображения (трехканальная)'''

        #tensor_mask_logit = F.interpolate(input=tensor_mask_logit, size=output_size, mode='bilinear', align_corners=False)
        # numpy_classes - каждый пиксел имеет значение, равное индексу наиболее вероятного по мнению модели классу
        numpy_mask_classes = torch.argmax(tensor_mask_logit.squeeze(), dim=0).detach().cpu().numpy()
        numpy_mask_rgb = NeuralNetwork.decode_segmap(numpy_mask_classes, num_classes=21)
        return numpy_mask_rgb


    @staticmethod
    def decode_segmap(numpy_mask_classes: np.ndarray, num_classes: int = 21) -> np.ndarray:
        '''Статический метод преобразования одноканальной маски изображения numpy_mask_classes, каждый пиксел которой
        содержит номер класса, в трехканальное изображение, каждый пиксел которого представлен 
        тремя значениями, кодирующими цвет (т.е. номера классов переводим в цвета)
        Входные параметры:
        numpy_mask_classes: np.ndarray - маска изображения (одноканальная) с номерами классов
        num_classes: int - общее количество классов в маске
        Возвращаемые значения:
        numpy_mask_rgb: np.ndarray - выходная маска изображения (трехканальная) - каждый класс закодирован своим цветом'''

        label_colors = np.array([(0, 0, 0),  # 0=background
        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        red   = np.zeros_like(numpy_mask_classes).astype(np.uint8)
        green = np.zeros_like(numpy_mask_classes).astype(np.uint8)
        blue  = np.zeros_like(numpy_mask_classes).astype(np.uint8)
        for num_class in range(0, num_classes):
            idx = numpy_mask_classes == num_class
            red[idx]   = label_colors[num_class, 0]
            green[idx] = label_colors[num_class, 1]
            blue[idx]  = label_colors[num_class, 2]
        numpy_mask_rgb = np.stack([red, green, blue], axis=2)
        return numpy_mask_rgb