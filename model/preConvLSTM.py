import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from torch import Tensor
import torch.nn as nn
import torch

from model.base.myConvLstm import ConvLSTM
from model.base.Resnet50 import ResModel



class PreConvLSTM(nn.Module):
    def __init__(self, classes=6, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.classes = classes

        self.encoder = ConvLSTM(3, 64, batch_first=True)
        self.decoder = ResModel(classes=classes)
        self.dropout = nn.Dropout()

    def forward(self, inputs : Tensor) -> Tensor:
        # inputs [batch, seq_len, c, h, w]
        if self.batch_first:
            inputs.permute(1, 0, 2, 3, 4)

        x = self.encoder(inputs)
        x = self.dropout(x)
        x = self.decoder(x)

        print(x.shape)
        return x

if __name__ == '__main__':
    model = PreConvLSTM()
    x = torch.zeros(8, 10, 3, 224, 224)
    model(x)