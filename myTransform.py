from torch import Tensor
from img_process import processImg
from model.Inception_resnetv2 import Inception_ResNetv2
from torchvision import transforms
from PIL import Image
from config import CLASSES


class MyTransform():
    def __init__(self) -> None:
        self.transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, .406],[0.229, 0.224, 0.225])
                                            ])

    def __call__(self, values):
        if isinstance(values, str):
            values = Image.open(values)
        if isinstance(values, Tensor):
            return values

        values = self.transform(values)
        if len(values.shape) == 3:
            values = values.unsqueeze(0)
        return values