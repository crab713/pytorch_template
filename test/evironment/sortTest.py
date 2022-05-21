import torch
import torch.nn as nn
import random
from tqdm import tqdm
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.lstm import LSTM


class SortModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LSTM(64, 256, 2, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(512, 64)
    
    def forward(self, inputs)->torch.Tensor:
        # input [batch, seq_len, 64]
        output,_ = self.encoder(inputs)
        output = self.dense(output)
        # output [batch, seq_len, 64]
        return output


def encoding(array):
    output = np.zeros((len(array), 64))
    for i in range(len(array)):
        output[i][array[i]] = 1
    return output

def getLabel(array:list):
    sortArray = sorted(array)
    return sortArray

model = SortModel().cuda()
# model = torch.load('sort.pth').cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# ---------------------
# train
# ---------------------
for i in tqdm(range(5000)):
    inputs = []
    labels = []
    for j in range(64):
        seq_len = 5
        array = []
        for _ in range(seq_len):
            array.append(random.randint(1, 63))

        x = encoding(array)
        label = getLabel(array)
        inputs.append(x)
        labels.append(label)
    inputs = torch.FloatTensor(inputs).cuda()    
    labels = torch.LongTensor(labels).cuda()

    outputs = model(inputs)
    outputs = outputs.permute(0,2,1)

    loss = criterion(outputs, labels)
    
    # if i%128 == 0:
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (i+1)%500 == 0:
        print()
        print('array:', torch.argmax(inputs[0], dim=1))
        print('real:', end='')
        print(labels[0])
        print('predict:', end='')
        print(torch.argmax(outputs[0], dim=0))
        print('loss:', loss.item())

torch.save(model, 'sort.pth')