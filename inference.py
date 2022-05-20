from typing import Union
import PIL
from torch import Tensor
from model.Inception_resnetv2 import Inception_ResNetv2
import torch
from torchvision import transforms
from PIL import Image
from dataset.base.myTransform import MyTransform


class OutputProcess():
    def __init__(self) -> None:
        self.classes = None
        assert self.classes is not None, '推理无类别'

    def __call__(self, output:Tensor) -> list:
        result = self.trans(output)
        return result

    def trans(self, inputs:Tensor) -> list:
        _, indices = inputs.topk(3)
        indices = indices.flatten()

        assert len(indices.shape) == 1
        result = [self.classes[i] for i in indices]
        return result

class Inferencer:
    def __init__(self, checkpoint_dir, classes=5, device='cpu') -> None:
        self.classes = classes
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        self.model = Inception_ResNetv2(classes=self.classes)
        self.load_model(self.model, checkpoint_dir)
        self.model.to(device)

        self.transform = MyTransform()
        self.outputProcess = OutputProcess()

    def __call__(self, values) -> list:
        return self.inference(values)

    def load_model(self, model, checkpoint_dir="output/ocr.model.epoch.2"):
        checkpoint = torch.load(checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        print("{} loaded!".format(checkpoint_dir))
    
    def inference(self, values:Union[str, Tensor]) -> list:
        self.model.eval()
        with torch.no_grad():
            inputs = self.transform(values)
            assert inputs.shape[1] == 3 and len(inputs.shape) == 4, 'inputs shape error, inputs:'+str(inputs.shape)

            output = self.model(inputs)

            result = self.outputProcess(output)
            return result


if __name__ == '__main__':
    inferencer = Inferencer('checkpoint\car.model.epoch.44', classes=242)
    output = inferencer('data/9_8.jpg')
    print(output)