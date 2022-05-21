from model.elimator import Elimator
import torch


model = Elimator(num_classes=4).cuda()
inputs = torch.ones(1, 3, 512, 512).cuda()
outputs = model(inputs)
wait = input()
print(outputs.shape)