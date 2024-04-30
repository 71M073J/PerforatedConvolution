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
    tf_test = [transforms.ToTensor(), ]
    data = "cifar"
    dataset1, dataset2, dataset3 = None, None, None
    g = None
    bs = 64
    if augment:
        tf.extend([transforms.RandomChoice([transforms.RandomCrop(size=32, padding=4),
                                            transforms.RandomResizedCrop(size=32)
                                            ]),
                   transforms.RandomHorizontalFlip()])
    if data == "cinic":
        tf.append(transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                       [0.24205776, 0.23828046, 0.25874835]))
        tf_test.append(transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                            [0.24205776, 0.23828046, 0.25874835]))
        tf = transforms.Compose(tf)
        tf_test = transforms.Compose(tf_test)
        dataset1 = DataLoader(CINIC10(partition="train", download=True, transform=tf),  # collate_fn=col,
                              num_workers=4, batch_size=bs, shuffle=True,
                              generator=g)
        dataset2 = DataLoader(
            CINIC10(partition="valid", download=True, transform=tf_test), num_workers=4,  # collate_fn=col,
            batch_size=bs, shuffle=True,
            generator=g, )
        dataset3 = DataLoader(
            CINIC10(partition="test", download=True, transform=tf_test), num_workers=4,  # collate_fn=col,
            batch_size=bs, shuffle=True,
            generator=g, )
    elif data == "cifar":

        tf.extend([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        tf_test.extend([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        tf = transforms.Compose(tf)
        tf_test = transforms.Compose(tf_test)
        dataset1 = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=tf), batch_size=bs, shuffle=True, num_workers=4)

        dataset2 = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=tf_test), batch_size=bs, shuffle=False,
            num_workers=4)

    net = torchvision.models.resnet18()
    op = torch.optim.SGD(net.parameters(), momentum=0.9, lr=0.1, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(op, [100, 150, 175], gamma=0.1)
    # op = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    epochs = 200
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=epochs)
    from Architectures.resnet import resnet18
    testnet = resnet18(num_classes=10, perforation_mode=(1,1))
    testnet._set_perforation((2,2))
    perf = eval_mode = [(1,1)]*15
    net = resnet18(num_classes=10, perforation_mode=perf)
    op = torch.optim.SGD(net.parameters(), momentum=0.9, lr=0.1, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(op, [100, 150, 175], gamma=0.1)
    #op = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    epochs = 200
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=epochs)
    #eval_mode=(2,2)
    with open(f"./resnet_perf_long_test{perf[0]}x{perf[1]}_out_eval{eval_mode[0]}x{eval_mode[1]}.txt", "w") as f:
        test_net(net, batch_size=bs, epochs=epochs, do_profiling=False, summarise=False, verbose=False,
                 make_imgs=False, plot_loss=True, vary_perf=None, file=f, eval_mode=eval_mode,
                 run_name=f"long_resnet18_perf_test{perf[0]}x{perf[1]}_out_eval{eval_mode[0]}x{eval_mode[1]}", dataset=dataset1, dataset2=dataset2, dataset3=dataset3, op=op,
                 lr_scheduler=lr_scheduler, validate=False if data == "cifar" else True)
