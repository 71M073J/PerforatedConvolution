import torch
import torch.nn as nn
import random

import torchvision.models
from pytorch_cinic.dataset import CINIC10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from main import test_net
nu = 0
g = torch.Generator()
def seed_worker(worker_id):
    #worker_seed = torch.initial_seed() % 2 ** 32

    global nu
    nu += 1
    np.random.seed(nu)
    random.seed(nu)
    g.manual_seed(nu)
    torch.manual_seed(nu)
    # print("Seed worker called YET AGAIN")

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    tf = transforms.Compose([transforms.ToTensor()])

    bs = 128
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
    test_net(net, batch_size=bs, epochs=500, do_profiling=False, summarise=False, verbose=False,
             make_imgs=False, plot_loss=False, vary_perf=None, file=None, eval_mode="none",
             run_name="long_resnet18_test", dataset=dataset1, dataset2=dataset2, dataset3=dataset3, )