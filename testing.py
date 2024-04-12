import time

from Architectures.resnet import resnet18
from Architectures.mobilenetv2 import MobileNetV2
from Architectures.mobilenetv3 import mobilenet_v3_small
import torch
import torch.nn as nn
a = nn.Conv2d(32,32,3, padding=2, padding_mode="reflect", device="cuda:0")
b = nn.Conv2d(32,32,3, padding=2, padding_mode="zeros", device="cuda:0")
tt = torch.randn((8, 32, 32, 32), device="cuda:0")
for i in range(100):
    a(tt)
    b(tt)

t1 = time.time()
for i in range(1000):
    a = nn.Conv2d(32, 32, 3, padding=2, padding_mode="reflect", device="cuda:0")
print(time.time() - t1, "seconds")
