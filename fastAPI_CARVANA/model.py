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


class NeuralNetwork(nn.Module):
    '''Класс для создания работы с нейронной сетью для семантической сегментации Carvana'''

    def __init__(self, device: str, path_to_model: str):
        '''Конструктор класса
        Входные параметры:
        device: str - устройство, на котором будет выполняться модель
        path_to_model: str - путь до сохраненной модели
        Возвращаемые значения: 
        объект класса NeuralNetwork'''

        super(NeuralNetwork, self).__init__()

        self.device = device
        self.model = smp.DeepLabV3Plus(encoder_name='timm-mobilenetv3_small_100', encoder_depth=5, encoder_weights='imagenet', 
                          encoder_output_stride=16, decoder_channels=256, decoder_atrous_rates=(12, 24, 36), 
                          in_channels=3, classes=1, activation=None, upsampling=4, aux_params=None).to(self.device)
        self.model.load_state_dict(torch.load(path_to_model))
        self.transformer = A.Compose([
            A.Resize(1024, 2048, cv2.INTER_AREA),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
            ToTensorV2(),
            ])

    
    def predict(self, tensor_image: torch.Tensor) -> torch.Tensor:
        '''Метод предсказания маски для изображения
        Входные параметры:
        tensor_image: torch.Tensor - изображение в тензорном формате
        Возвращаемые значения:
        tensor_mask_logit: torch.Tensor - предсказанная маска в тензорном формате'''

        self.model.eval()    
        tensor_mask_logit = self.model(tensor_image)
        return tensor_mask_logit


    def preprocessing(self, pil_image: PIL.Image) -> torch.Tensor:
        '''Метод предобработки данных перед подачей в сеть
        Входные параметры:
        pil_image: PIL.Image - изображение, для которого нужно предсказать маску
        Возвращаемые значения:
        tensor_image: torch.Tensor - подготовленное для подачи в сеть изображение в тензорном формате, 
        #для которого нужно предсказать маску '''

        np_image = np.asarray(pil_image).astype('float')/255.0
        transformed = self.transformer(image=np_image)
        tensor_image = transformed['image']
        tensor_image = tensor_image.unsqueeze(0)
        tensor_image = tensor_image.to(self.device).float()
        return tensor_image


    @staticmethod
    def postprocessing(tensor_mask_logit: torch.Tensor, output_size: tuple=(1280, 1918), mask_treashold: float=0.5) -> PIL.Image:
        '''Статический метод постобработки данных, полученных с выхода сети
        Входные параметры:
        tensor_mask_logit: torch.Tensor - предсказанная сетью маска в тензорном формате в масштабе logit 
        output_size: tuple - пространственная размерность, к которой нужно привести выходные маски
        mask_treashold: float - порог, по которому будет определяться класс каждой точки для масок
        Возвращаемые значения:
        pil_image: PIL.Image - выходная маска изображения'''

        tensor_mask_logit = F.interpolate(input=tensor_mask_logit, size=output_size, mode='bilinear', align_corners=False)
        tensor_mask_prob = torch.sigmoid(tensor_mask_logit)
        tensor_mask = torch.where(tensor_mask_prob > mask_treashold, 1, 0)
        numpy_mask = (tensor_mask[0].cpu().numpy() * 255.0)[0] # [0] - избавляемся от батч размерности и от канальной размерности
        pil_image = PIL.Image.fromarray(numpy_mask.astype('uint8'), 'L')
        return pil_image