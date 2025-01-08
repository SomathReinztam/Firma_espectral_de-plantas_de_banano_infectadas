"""
1_conv1d.py

"""

import torch
import torch.nn as nn
import numpy as np

print()
print('input')
print()
x = np.linspace(0, 6, 5)
x = np.reshape(x, (1, 5))
x = torch.from_numpy(x).float()
x = x.unsqueeze(1)
print(x)
print(x.shape)
print()

print('conv1: in_channels=1, out_channels=3, kernel_size=1')
conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=1)
print()

x = conv(x)
print(x)
print(x.shape)
print()

print('conv2: in_channels=3, out_channels=1, kernel_size=1')
conv2 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
print()

x = conv2(x)
print(x)
print(x.shape)
print()