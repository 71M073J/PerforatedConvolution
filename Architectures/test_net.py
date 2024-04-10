import torch.nn as nn
import torch
import torchvision
from torchvision import transforms

from Architectures.PerforatedConv2d import PerforatedConv2d
from main import test_net

class Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kern1, kern2, kern3, pads=("same", "same", "same"),
                 down=False, dropout=0.1):
        super().__init__()
        self.a_attn = PerforatedConv2d(in_channels, out_channels, 1, padding=pads[2])
        self.bnx = nn.BatchNorm2d(out_channels)
        self.c1 = PerforatedConv2d(in_channels, mid_channels, kern1, padding=pads[0])
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.c2 = PerforatedConv2d(mid_channels, mid_channels, kern2, padding=pads[1])
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.c3 = PerforatedConv2d(mid_channels, out_channels, kern3, padding=pads[2])
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.p3 = pads[2]
        self.drop = nn.Dropout(p=dropout)
        self.down = down
        if down:
            self.down1 = PerforatedConv2d(out_channels, out_channels, 7, stride=2, padding="same")  # 16
            self.bnd = nn.BatchNorm2d(out_channels)
            self.hs = nn.Hardswish()

    def forward(self, x):
        x1 = self.relu(self.bn1(self.c1(x)))
        x1 = self.relu(self.bn2(self.c2(x1)))
        x1 = (self.bn3(self.c3(x1)))
        if self.p3 == 0:
            x1 = self.relu(x1)
        else:
            x1 = self.relu(x1 + self.bnx(self.a_attn(x)))
        if self.down:
            x1 = self.hs(self.bnd(self.down1(x1)))
        return self.drop(x1)


class TestNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[Block(3, 64, 64, 5, 5, 5, down=True),
                                      Block(64, 128, 128, 5, 5, 5, down=True),  # 16
                                      #Block(64, 128, 64, 5, 5, 5),
                                      Block(128, 256, 256, 5, 5, 5, down=True),  # 8
                                      #Block(128, 256, 128, 5, 5, 5),
                                      #Block(128, 256, 256, 5, 5, 5, down=True),  # 4
                                      Block(256, 256, 256, 5, 5, 4, pads=("same", "same", 0))])  # 1
        self.ls = nn.Sequential(*[nn.Linear(256, 256), nn.Hardswish(), nn.Linear(256, num_classes)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.ls(torch.flatten(x, start_dim=1))


if __name__ == "__main__":
    from Architectures.resnet import resnet18
    from Architectures.mobilenetv3 import mobilenet_v3_small
    net = TestNet().cuda()
    #net = resnet18(num_classes=10, perforation_mode=(2,2))
    op = torch.optim.Adam(net.parameters(), lr=0.001)
    augment = True
    tf = [transforms.ToTensor(), ]
    tf_test = [transforms.ToTensor(), ]
    data = "cifar"
    g = None
    bs = 64
    tf.extend([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    tf_test.extend([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    tf.extend([transforms.RandomChoice([transforms.RandomCrop(size=32, padding=4),
                                        transforms.RandomResizedCrop(size=32)
                                        ]),
               transforms.RandomHorizontalFlip()])
    tf = transforms.Compose(tf)
    tf_test = transforms.Compose(tf_test)
    dataset1 = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=tf), batch_size=bs, shuffle=True, num_workers=4)

    dataset2 = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=tf_test), batch_size=bs, shuffle=False,
        num_workers=4)
    test_net(net, run_name="testing", verbose=False, dataset=dataset1, dataset2=dataset2, summarise=True, op=op)
