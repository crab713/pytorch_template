print('wocao')
import torch
from torch import nn

conv1 = nn.Conv2d(1, 5, 3, padding=1)

x = torch.FloatTensor(1,1,100,100)
print(conv1(x).shape)