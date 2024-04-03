import torch
import torch.nn as nn
import random

import torchvision.models
from pytorch_cinic.dataset import CINIC10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from main import test_net

if __name__ == "__main__":
    # np.random.seed(0)
    # random.seed(0)
    augment = True
    tf = [transforms.ToTensor(), ]
    if augment:
        tf.extend([transforms.RandomResizedCrop(size=32), transforms.RandomHorizontalFlip])
    tf.append(transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                   [0.24205776, 0.23828046, 0.25874835]))
    tf = transforms.Compose(tf)
    g = None
    bs = 64
    dataset1 = DataLoader(CINIC10(partition="train", download=True, transform=tf),  # collate_fn=col,
                          num_workers=4, batch_size=bs, shuffle=True,
                          generator=g)
    dataset2 = DataLoader(
        CINIC10(partition="valid", download=True, transform=tf), num_workers=4,  # collate_fn=col,
        batch_size=bs, shuffle=True,
        generator=g, )
    dataset3 = DataLoader(
        CINIC10(partition="test", download=True, transform=tf), num_workers=4,  # collate_fn=col,
        batch_size=bs, shuffle=True,
        generator=g, )
    net = torchvision.models.resnet18()
    op = torch.optim.SGD(net.parameters(), momentum=0.9, lr=0.1, nesterov=True, weight_decay=0.0001)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(op, [100, 150, 175], gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=300)
    with open("./resnet_long_test_out.txt", "w") as f:
        test_net(net, batch_size=bs, epochs=300, do_profiling=False, summarise=False, verbose=False,
                 make_imgs=False, plot_loss=True, vary_perf=None, file=f, eval_mode=None,
                 run_name="long_resnet18_test", dataset=dataset1, dataset2=dataset2, dataset3=dataset3, op=op,
                 lr_scheduler=lr_scheduler)
