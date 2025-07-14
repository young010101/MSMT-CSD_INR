import torch

from torch import nn

net = nn.Sequential(nn.Linear(2, 1))

print(net((1, 2)))