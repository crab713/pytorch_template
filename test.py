import torch

x = torch.zeros(20, 64)
y = torch.zeros(64, 256)
z = torch.zeros(1, 256)

output = x @ y + z

print(output.shape)