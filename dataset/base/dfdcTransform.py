from typing import Tuple
from torch import Tensor
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import cv2

class DFDCTransform():
    def __init__(self, height = 224, width = 224) -> None:
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((height, width)),
                                            transforms.Normalize([0.485, 0.456, .406],[0.229, 0.224, 0.225])
                                            ])
        self.height = height
        self.width = width

    def __call__(self, values : np.ndarray) -> Tensor:
        # value [frame, h, w, c]
        if isinstance(values, Tensor):
            return values
        if len(values.shape) != 4:
            raise RuntimeError(
                'values has inconsistent shape: got {}, expect length 4'
                    .format(values.shape))
        

        (frames_num, h, w, c) = values.shape
        outputs = torch.zeros((frames_num, c, self.height, self.width))
        for i in range(frames_num):
            frame = self.transform(values[i])
            outputs[i] = frame
        
        return outputs # [frames_num, c, h, w]


if __name__ == '__main__':
    trans = DFDCTransform()
    x = np.zeros((16, 720, 1280, 3))
    output = trans(x)
    print(output.shape)