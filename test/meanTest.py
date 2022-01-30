import torch
import torch.nn as nn
import random
from tqdm import tqdm
import numpy as np
from model.lstm import LSTM


class SortModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = LSTM(64, 256, 2, batch_first=True)
        self.dense = nn.Linear(256, 64)
    
    def forward(self, inputs)->torch.Tensor:
        # input [batch, seq_len, 64]
        output,(h_n, c_n) = self.encoder(inputs)
        output = self.dense(output)
        # output [batch, seq_len, 2]
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# ---------------------
# train
# ---------------------
for i in tqdm(range(10000)):
    inputs = []
    labels = []
    for j in range(64):
        seq_len = 5
        array = []
        for _ in range(seq_len):
            array.append(random.randint(1, 63))
        seq_mean = int(sum(array) / len(array))
        labels.append(seq_mean)
        inputs.append(encoding(array))
    inputs = torch.FloatTensor(inputs).cuda()    
    labels = torch.LongTensor(labels).cuda()

    outputs = model(inputs)
    outputs = outputs.permute(0,2,1)[:,:,-1]
    # print(outputs.shape)
    # print(labels.shape)
    # exit()
    loss = criterion(outputs, labels)
    
    # if i%128 == 0:
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (i+1)%500 == 0:
        print()
        print('array:', torch.argmax(inputs[1], dim=1))
        print('real:', end='')
        print(labels[1])
        print('predict:', end='')
        print(torch.argmax(outputs[1], dim=0))
        print('loss:', loss.item())

torch.save(model, 'sort.pth')