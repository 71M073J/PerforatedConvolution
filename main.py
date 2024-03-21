import copy
import os.path
import time
from typing import Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing_extensions import Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from contextlib import ExitStack
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary

from Architectures.PerforatedConv2d import PerforatedConv2d


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    print("Seed worker called YET AGAIN")


g = torch.Generator()
g.manual_seed(0)
torch.manual_seed(0)
import random

np.random.seed(0)
random.seed(0)


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class Cifar10CnnModel(ImageClassificationBase):

    def __init__(self, perf="none", interpolate_grad=True):
        super().__init__()
        self.network = nn.Sequential(
            PerforatedConv2d(3, 32, kernel_size=3, padding=1, perforation_mode=perf, interpolate_grad=interpolate_grad),
            nn.ReLU(),
            PerforatedConv2d(32, 64, kernel_size=3, stride=1, padding=1, perforation_mode=perf,
                             interpolate_grad=interpolate_grad),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

            PerforatedConv2d(64, 128, kernel_size=3, stride=1, padding=1, perforation_mode=perf,
                             interpolate_grad=interpolate_grad),
            nn.ReLU(),
            PerforatedConv2d(128, 128, kernel_size=3, stride=1, padding=1, perforation_mode=perf,
                             interpolate_grad=interpolate_grad),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

            PerforatedConv2d(128, 256, kernel_size=3, stride=1, padding=1, perforation_mode=perf,
                             interpolate_grad=interpolate_grad),
            nn.ReLU(),
            PerforatedConv2d(256, 256, kernel_size=3, stride=1, padding=1, perforation_mode=perf,
                             interpolate_grad=interpolate_grad),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, xb):
        return self.network(xb)


from PIL import Image


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


class CINIC10(Dataset):

    def __init__(self, dir="./CINIC-10", mode="train", ):
        if mode not in ["train", "test", "valid"]:
            print("Invalid Dataset mode:", mode)
            raise NotADirectoryError
        self.path = os.path.join(dir, mode)
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.imgpaths = []
        for cl in self.classes:
            classpaths = os.listdir(os.path.join(self.path, cl))
            for img in classpaths:
                if img.endswith(".png"):
                    self.imgpaths.append(os.path.join(cl, img))

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, item):
        img = torch.tensor(((load_image(os.path.join(self.path, self.imgpaths[item])) / (255 / 2)) - 1.),
                           dtype=torch.float32)
        label = torch.tensor(self.classes.index(self.imgpaths[item].split("\\")[0]))
        if len(img.shape) != 3:
            img = torch.stack((img, img, img)).transpose(0, 2)
        if img.shape[2] == 3:
            img = img.transpose(0, 2)
        elif img.shape[1] == 3:
            img = img.transpose(0, 1)
        return img, label


def col(batch):
    return torch.stack([x[0] for x in batch]), torch.stack([x[1] for x in batch])


def compare_speed():
    device = "cuda:0"
    timing = "forward"
    # device = "cpu"
    in_channels = 64
    backwards = True
    for in_channels in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        l1 = nn.Conv2d(in_channels, in_channels * 2, 5).to(device)
        l2 = PerforatedConv2d(in_channels, in_channels * 2, 5, perforation_mode="both").to(device)
        l3 = PerforatedConv2d(in_channels, in_channels * 2, 5, perforation_mode="first").to(device)
        l4 = PerforatedConv2d(in_channels, in_channels * 2, 5, perforation_mode="second").to(device)
        l5 = PerforatedConv2d(in_channels, in_channels * 2, 5, perforation_mode="none").to(device)
        ls = [l1, l2, l3, l4, l5]
        os = [torch.optim.SGD(l.parameters(), lr=.001) for l in ls]
        total_times = [0, 0, 0, 0, 0]
        times = [[], [], [], [], []]
        # with torch.no_grad():
        iters = 100
        t0 = 0
        for i in range(iters):
            input_vec = torch.randn(size=(4096 // in_channels, in_channels, 64, 64), requires_grad=True).to(device)
            for j, (l, o) in enumerate(zip(ls, os)):
                o.zero_grad()
                if j > 0:
                    print(l.perforation)
                torch.cuda.current_stream().synchronize()
                t0 = time.time()
                out1 = l(input_vec)
                if backwards:
                    loss = out1.sum()
                    loss.backward()
                    o.step()
                torch.cuda.current_stream().synchronize()
                t1 = time.time() - t0
                if i > 2:
                    total_times[j] += t1
                    times[j].append(t1)
            if i % 10 == 0:
                print(i)

                # TODO:
                # convergence time
                # backward pass time
                # final accuracy when converged
                # loss plot

                # softmax confidence/entropy of the output probability vector
                # try resnet as well, and other NN architectures
                # mobilenet with CINIC10 dataset
                # also try larger image dataset - maybe fruits?

        import matplotlib.pyplot as plt

        print(f"Normal convLayer: average of {total_times[0] / len(times[0])} seconds")
        for i in range(4):
            print(
                f"Perforated convLayer: {ls[i + 1].perforation}: average of {total_times[i + 1] / len(times[i + 1])} seconds")
        plt.plot(times[0], label=f"Normal Conv, {int(1000 * (total_times[0] / len(times[0]) * 1000)) / 1000}ms avg")
        for i in range(4):
            plt.plot(times[i + 1],
                     label=f"PerfConv {ls[1 + i].perforation}, {int(1000 * (total_times[i + 1] / len(times[i + 1]) * 1000)) / 1000}ms avg")
        plt.legend()
        plt.ylabel("Time elapsed (s)")
        if backwards:
            plt.savefig(f"in_channels_{in_channels}_with_backward.png")
        else:
            plt.savefig(f"in_channels_{in_channels}.png")
        plt.show()
        plt.clf()


def test_net(net, batch_size=128, verbose=False, epochs=10, summarise=False, run_name="", do_profiling=False,
             make_imgs=False, test_every_n=5, plot_loss=False, report_class_accs=False, vary_perf=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bs = batch_size
    # transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset = DataLoader(AppleDataset(mode="train"), num_workers=4, batch_size=bs, shuffle=True)
    dataset = DataLoader(CINIC10(mode="train"), collate_fn=col,
                         num_workers=4, batch_size=bs, shuffle=True, worker_init_fn=seed_worker,
                         generator=g)
    dataset2 = DataLoader(
        CINIC10(mode="test"), num_workers=4, collate_fn=col,
        batch_size=bs, shuffle=True, worker_init_fn=seed_worker,
        generator=g, )

    # net = simpleNN(perf="both", interpolate_gradients=True)
    # net = Cifar10CnnModel(perf="both", interpolate_grad=True)
    loss_fn = nn.CrossEntropyLoss()
    if summarise:
        summary(net, input_size=(bs, 3, 32, 32))

    net.to(device)
    op = optim.Adam(net.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(op, 'min')
    # op = optim.SGD(net.parameters(), lr=0.001)
    n_epochs = epochs
    items = len(dataset)
    testitems = len(dataset2)
    # TODO TRAIN TEST VALIDATE SPLIT

    losses = []
    ep_losses = []
    test_losses = []
    ep_test_losses = []
    #
    #
    # time.sleep(123)
    img = 0
    n_conv = len(net.perforation)
    if make_imgs:
        if type(net) == MobileNetV2:
            img = list(net.children())[0][0][0].weight.grad.view(
                list(net.children())[0][0][0].weight.shape[0], -1).detach().cpu()

        if type(net) == MobileNetV3:
            img = list(net.children())[0][0][0].weight.grad.view(
                list(net.children())[0][0][0].weight.shape[0], -1).detach().cpu()
        elif type(net) == ResNet:
            img = list(net.children())[0].weight.grad.view(list(net.children())[0].weight.shape[0],
                                                           -1).detach().cpu()
    for epoch in range(n_epochs):
        l = 0
        accs = 0
        networkCallTime = 0
        class_accs = np.zeros((2, 15))
        with ExitStack() as stack:
            if do_profiling:
                prof = stack.enter_context(torch.autograd.profiler.profile(profile_memory=True, use_cuda=True))
            for i, (batch, classes) in enumerate(dataset):
                if make_imgs and i % 100 == 0:
                    if type(net) == MobileNetV2 or type(net) == MobileNetV3:
                        img = list(net.children())[0][0][0].weight.grad.view(
                            list(net.children())[0][0][0].weight.shape[0], -1).detach().cpu()
                    elif type(net) == ResNet:
                        img = list(net.children())[0].weight.grad.view(list(net.children())[0].weight.shape[0],
                                                                 -1).detach().cpu()
                if vary_perf is not None:
                    if vary_perf == "incremental":
                        randn = np.random.randint(0, n_conv)
                        perfs = (["both"] * randn) + (["none"] * (n_conv - randn))
                    elif vary_perf == "random":
                        perfs = np.array(["both"] * n_conv)
                        perfs[np.random.random(n_conv) > 0.5] = "none"
                    net._set_perforation(perfs)
                batch = batch.to(device)
                t0 = time.time()
                pred = net(batch)
                if i > 5:
                    networkCallTime += time.time() - t0
                loss = loss_fn(pred, classes.to(device))
                loss.backward()
                acc = (F.softmax(pred.detach().cpu(), dim=1).argmax(dim=1) == classes)
                for clas in classes[acc]:
                    class_accs[0, clas] += 1
                for clas in classes:
                    class_accs[1, clas] += 1
                l += loss.detach().cpu()
                losses.append(loss.detach().cpu())
                accs += acc.sum()

                # if i % 256 == 0:
                #    #with torch.no_grad():
                #        plt.plot(net.relu7(torch.arange(-60, 60, 0.1).cuda()).detach().cpu())
                #        plt.show()
                if make_imgs:
                    if type(net) == MobileNetV2 or type(net) == MobileNetV3:
                        img += list(net.children())[0][0][0].weight.grad.view(
                            list(net.children())[0][0][0].weight.shape[0], -1).detach().cpu()
                    elif type(net) == ResNet:
                        img += list(net.children())[0].weight.grad.view(list(net.children())[0].weight.shape[0],
                                                                   -1).detach().cpu()
                if i % 100 == 0:
                    if make_imgs:  # todo maybe sum/avg of gradients over 100 batches instead of just 100th batch

                        im = plt.imshow(img)
                        values = [torch.min(img), torch.max(img)]
                        colors = [im.cmap(im.norm(value)) for value in values]
                        # create a patch (proxy artist) for every color
                        patches = [mpatches.Patch(color=colors[i], label=[f"Min: {str(values[i])[7:-1]}",
                                                                          f"Max: {str(values[i])[7:-1]}"][i]) for i in
                                   range(len(values))]
                        # put those patched as legend-handles into the legend
                        plt.legend(handles=patches, bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.0)
                        plt.tight_layout()
                        newpath = f"./imgs/{run_name}"
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)
                        plt.savefig(os.path.join(newpath, f"e{epoch}-b{i}.png"))
                        # plt.show()
                        plt.clf()
                if True:
                    # with torch.no_grad():
                    #    plt.plot(np.arange(-15, 15, 0.01), net.relu7(torch.arange(-15, 15, 0.01, dtype=torch.float, device=device)).detach().cpu())
                    #    plt.savefig(f"func{i}.png")
                    #    plt.clf()
                    op.step()
                    op.zero_grad()
                    # l /= (64 // bs)
                    # batchacc = accs / (bs * (64 // bs))
                    batchacc = accs / bs
                    if i % 100 == 0:
                        if verbose:
                            print(
                                f"Loss: {loss.detach().cpu()}, batch acc:{batchacc} progress: {int((i / items) * 10000) / 100}%, Batch",
                                i)
                            print("Class accs:", class_accs[0] / (class_accs[1] + 1e-12))
                            print(int((time.time() - t0) * 100) / 100, "Seconds elapsed for 1 batch forward.",
                                  "Average net call:", int((networkCallTime / (i + 1)) * 100000) / 100, "Milliseconds")

                        if do_profiling:
                            stack.close()
                            print(prof.key_averages().table(row_limit=10))
                            with open("all_nets_profiling.txt", "a") as f:
                                print("network", run_name, ":", file=f)
                                print(prof.key_averages().table(row_limit=10), file=f)
                            return
                    # l = 0
                    accs = 0

            scheduler.step(l.item() / i)
            ep_losses.append(l.item()/i)
            losses.append(np.nan)
            print("Average Epoch Train Loss:", l.item() / i)

        if (epoch % test_every_n == (test_every_n - 1)) or plot_loss:
            net.eval()
            if vary_perf is not None:
                net._set_perforation(["both"] * n_conv)
            class_accs = np.zeros((2, 15))
            l2 = 0
            for i, (batch, classes) in enumerate(dataset2):

                pred = net(batch.to(device))
                loss = loss_fn(pred, classes.to(device))
                l2 += loss.detach().cpu()
                test_losses.append(loss.detach().cpu())
                acc = (F.softmax(pred.detach().cpu(), dim=1).argmax(dim=1) == classes)
                for clas in classes[acc]:
                    class_accs[0, clas] += 1
                for clas in classes:
                    class_accs[1, clas] += 1
                if i % 50 == 0 and verbose:
                    print(
                        f"Loss: {loss.detach().cpu().item()}, batch acc:{acc.sum() / bs} progress: {int((i / testitems) * 10000) / 100}%")

                    print("Class accs:", class_accs[0] / (class_accs[1] + 1e-12))
            test_losses.append(np.nan)
            if report_class_accs:
                print("Class accs:", class_accs[0] / (class_accs[1] + 1e-12))
            print("Average Epoch Test Loss:", l2.item() / i)
            with open("./out.txt", "a") as f:
                print(
                    f"Epoch {epoch}, network {type(net).__name__}, perf mode {net.perforation}, gradient filtering {net.grad_conv}",
                    file=f)
                print(
                    f"Epoch {epoch}, network {type(net).__name__}, perf mode {net.perforation}, gradient filtering {net.grad_conv}")
                print("Class accs:", class_accs[0] / (class_accs[1] + 1e-12), file=f)
            ep_test_losses.append(l2.item()/i)
            net.train()
        # print("saving best model...")
        # torch.save(net.state_dict(), f"./model.pth")
        # torch.save(op.state_dict(), f"./op.pth")
    plt.scatter(range(len(losses)), losses, label="losses")
    plt.scatter(range(len(test_losses) + (len(test_losses) // epochs)),
             ([np.nan] * (len(test_losses) // epochs)) + test_losses, label="test losses")
    plt.plot(np.arange((len(losses) // epochs), len(losses) + (len(losses) // epochs), (len(losses) // epochs)), ep_losses, color="r", label="Avg epoch Train loss")
    plt.plot(np.arange((len(losses) // epochs), len(losses) + (len(losses) // epochs), (len(losses) // epochs)), ep_test_losses, color="g", label="Avg Epoch Test loss")
    plt.xticks(np.arange(0, len(losses) + (len(losses) // epochs), (len(losses) // epochs)), np.arange(0, epochs + 1, 1))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"loss_timeline_{run_name}.png")
    #plt.show()
    plt.clf()

if __name__ == "__main__":

    from Architectures.mobilenetv2 import MobileNetV2
    from Architectures.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small, MobileNetV3
    from Architectures.resnet import resnet152, resnet18, ResNet

    # from Architectures.test_resnet import resnet152

    nets = [
        # TODO check the article for interpolation strategies
        #  todo different ways of interpolation/perforation (from the article) measuere the
        #   effect on just one layer(implement in architecture probably)
        #   work on gradient comparison on one layer thing but use the entire network so we can do the comparison
        #   while learning (just maybe do matrix difference images or something like that,
        #   such as gradient number size comparison in a distibution
        #   dont bother much (or at all) with speedup, focus on "accuracy" performance

        # TODO treniranje do saturationa lossa - da dobimo hitrost konvergence pri treniranju (run on something not my laptop)

        # MMobileNetV2(num_classes=10, perforation_mode="both", use_custom_interp=True, grad_conv=True),
        #MobileNetV2(num_classes=10, perforation_mode="both", use_custom_interp=True, grad_conv=True),  # , dropout=0.0
        #MobileNetV2(num_classes=10, perforation_mode="both", use_custom_interp=True, grad_conv=False),
        # MobileNetV2(num_classes=10, perforation_mode="both", use_custom_interp=False),
        # MobileNetV2(num_classes=10, perforation_mode="none"),
        #mobilenet_v3_small(num_classes=10, perforation_mode="both", use_custom_interp=True, grad_conv=True),
        # mobilenet_v3_small(num_classes=10, perforation_mode="both", use_custom_interp=True, grad_conv=False),
        # mobilenet_v3_small(num_classes=10, perforation_mode="both", use_custom_interp=False),
        # mobilenet_v3_small(num_classes=10, perforation_mode="none"),
        # mobilenet_v3_large(num_classes=10, perforation_mode="both", use_custom_interp=True, grad_conv=True),
        # mobilenet_v3_large(num_classes=10, perforation_mode="both", use_custom_interp=True, grad_conv=False),
        # mobilenet_v3_large(num_classes=10, perforation_mode="both", use_custom_interp=False),
        # mobilenet_v3_large(num_classes=10, perforation_mode="none"),
        resnet18(num_classes=10, perforation_mode="both", use_custom_interp=True, grad_conv=True),
        resnet18(num_classes=10, perforation_mode="both", use_custom_interp=True, grad_conv=False),
        resnet18(num_classes=10, perforation_mode="both", use_custom_interp=False),
        resnet18(num_classes=10, perforation_mode="none"),
        # resnet152(num_classes=10, perforation_mode="both", use_custom_interp=True, grad_conv=True),
        # resnet152(num_classes=10, perforation_mode="both", use_custom_interp=True, grad_conv=False),
        # resnet152(num_classes=10, perforation_mode="both", use_custom_interp=False),
        # resnet152(num_classes=10, perforation_mode="none"),
    ]
    for net in nets:
        # TODO make profiler spit out more data
        # TODO run convergence tests on fri machine
        # TODO resnet
        t = time.time()
        vary_perf = "random"
        test_net(net, batch_size=128, epochs=2, do_profiling=False, summarise=False, verbose=True, make_imgs=False,
                 plot_loss=True, vary_perf=vary_perf,
                 run_name=type(net).__name__ +
                          (f"-Vary_perforation-{vary_perf}-Grad_conv-{str(net.grad_conv) if net.use_custom_interp else ''}-" if vary_perf is not None else ("" if net.perforation[0] == "none" else ("" + net.perforation[0] + (
                              ("-Grad_conv-" + str(net.grad_conv)) if net.use_custom_interp else "")))))
        print(
            f"{type(net)}, Perforation mode: {net.perforation}, Custom interpolation: {net.use_custom_interp}, {time.time() - t} seconds")

    # test_net(resnet152(num_classes=10, perforation_mode="both"), batch_size=32)

    # compare_speed()
    # quit()
    quit()
    h = PerforatedConv2d(3, 5, 3, stride=2, padding=2)
    sz = 10
    x = torch.arange(0, 3 * sz * sz, 1.0, requires_grad=True).reshape(1, 3, sz, sz)
    print(h.conv(x).shape)
    optimizer = torch.optim.SGD(h.parameters(), lr=0.01, momentum=0.9)
    test = h(x).sum()
    print(test)
    test.backward()

    quit()
