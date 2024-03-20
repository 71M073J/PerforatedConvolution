import torch.nn as nn
import torch
import time
if __name__ == "__main__":
    a = torch.ones((16, 3, 256, 256)).float()
    #b = torch.ones((16, 128, 64, 64)).float()
    conv1 = nn.Conv2d(3, 32, 3, (1, 1))
    conv2 = nn.Conv2d(3, 32, 3, (1, 2))
    sum = 0
    for i in range(1000):
        first = time.time()
        x = conv1(a[:, :, :, ::2])
        loss = x.sum()
        loss.backward()
        last = time.time()
        sum += last - first
        if i % 100 == 0:
            print(sum / (i + 1), "ms")
            print(x.shape)
        del x
    print(sum, "Milliseconds elapsed per call, not strided")
    sum = 0
    for i in range(1000):
        first = time.time()
        x = conv2(a)
        loss = x.sum()
        loss.backward()
        last = time.time()
        sum += last - first
        if i % 100 == 0:
            print(sum / (i + 1), "ms")
            print(x.shape)
        del x
    print(sum, "Milliseconds elapsed per call, strided")