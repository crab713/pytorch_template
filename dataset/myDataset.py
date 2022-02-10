from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.utils import shuffle
import os
from torchvision import transforms
from PIL import Image
import cv2
import pandas as pd
from dataset.base.myTransform import MyTransform

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_list = os.listdir(data_path)
        self.transform = MyTransform()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.data_path, self.image_list[i]))
        image = self.transform(image)
        if image.shape[0] != 3:
            print(os.path.join(self.data_path, self.image_list[i]))
            print(image.shape)
        label = self.image_list[i].split('_')[0]
        assert label != None , print(self.image_list[i])

        label = torch.tensor(int(label), dtype=torch.long)
        
        output = {"image": image,
                  "label": label}
        return output
