import os.path
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models.resnet
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.distributions import Categorical
from contextlib import ExitStack
from torchinfo import summary
from pytorch_cinic.dataset import CINIC10
from Architectures.PerforatedConv2d import PerforatedConv2d
from Architectures.mobilenetv2 import MobileNetV2
from Architectures.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small, MobileNetV3
from Architectures.resnet import resnet152, resnet18, ResNet
nu = 0
g = torch.Generator()


def seed_worker(worker_id):
    # worker_seed = torch.initial_seed() % 2 ** 32

    global nu
    nu += 1
    np.random.seed(nu)
    random.seed(nu)
    g.manual_seed(nu)
    torch.manual_seed(nu)
    # print("Seed worker called YET AGAIN")


import random

np.random.seed(0)
random.seed(0)
tf = transforms.Compose([transforms.ToTensor()])


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


from PIL import Image


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


class CINIC10_(Dataset):

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
    for in_channels in [2, 64, 256]:
        l1 = nn.Conv2d(in_channels, in_channels * 2, 5).to(device)
        l2 = PerforatedConv2d(in_channels, in_channels * 2, 5, perforation_mode="both", kind=True).to(device)
        l3 = PerforatedConv2d(in_channels, in_channels * 2, 5, perforation_mode="both", kind=False).to(device)
        # l3 = PerforatedConv2d(in_channels, in_channels * 2, 5, perforation_mode="first").to(device)
        l4 = PerforatedConv2d(in_channels, in_channels * 2, 5, perforation_mode="trip", kind=True).to(device)
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
                # try as well, and other NN architectures
                # mobilenet with CINIC10 dataset
                # also try larger image dataset - maybe fruits?

        import matplotlib.pyplot as plt

        print(f"Normal convLayer: average of {total_times[0] / len(times[0])} seconds")
        for i in range(len(ls) - 1):
            print(ls[i + 1].kind)
            print(
                f"Perforated convLayer: {ls[i + 1].perforation}: average of {total_times[i + 1] / len(times[i + 1])} seconds")
        plt.plot(times[0], label=f"Normal Conv, {int(1000 * (total_times[0] / len(times[0]) * 1000)) / 1000}ms avg")
        for i in range(len(ls) - 1):
            plt.plot(times[i + 1],
                     label=f"PerfConv {ls[1 + i].perforation} {'conv interp:' + str(ls[i + 1].kind) if ls[i + 1].perforation != 'none' else ''}, {int(1000 * (total_times[i + 1] / len(times[i + 1]) * 1000)) / 1000}ms avg")
        plt.legend()
        plt.ylabel("Time elapsed (s)")
        if backwards:
            plt.savefig(f"in_channels_{in_channels}_with_backward.png")
        else:
            plt.savefig(f"in_channels_{in_channels}.png")
        plt.show()
        plt.clf()


def train(do_profiling, dataset, n_conv, p, device, loss_fn, make_imgs, losses, op, verbose, file, items, epoch,
          ep_losses, vary_perf, eval_mode, net, bs, run_name):
    l = 0
    train_accs = []
    entropies = 0
    accs = 0
    networkCallTime = 0
    class_accs = np.zeros((2, 15))
    weights = []
    with ExitStack() as stack:
        if do_profiling:
            prof = stack.enter_context(torch.autograd.profiler.profile(profile_memory=True, use_cuda=True))
        for i, (batch, classes) in enumerate(dataset):

            if vary_perf is not None:
                if vary_perf == "incremental":
                    randn = np.random.randint(0, n_conv, size=2)  # TODO make this actually sensible
                    perfs = np.array(["trip"] * n_conv)
                    perfs[np.min(randn):np.max(randn)] = np.array(["both"] * (np.max(randn) - np.min(randn)))
                    perfs[np.max(randn):] = "none"
                elif vary_perf == "random":
                    rn = np.random.random(n_conv)
                    perfs = np.array(["both"] * n_conv)
                    perfs[rn > 0.66666] = np.array(["none"] * len(perfs[rn > 0.66666]))
                    perfs[rn < 0.33333] = np.array(["trip"] * len(perfs[rn < 0.33333]))
                net._set_perforation(perfs)
            if eval_mode is not None:
                net._set_perforation(p)
            batch = batch.to(device)
            t0 = time.time()
            pred = net(batch)
            acc = torch.sum(F.softmax(pred.detach().cpu(), dim=1).argmax(dim=1) == classes) / bs
            train_accs.append(acc)
            if i > 5:
                networkCallTime += time.time() - t0
            loss = loss_fn(pred, classes.to(device))
            loss.backward()
            if make_imgs:
                if type(net) == MobileNetV2 or type(net) == MobileNetV3:
                    weights.append(list(net.children())[0][0][0].weight.grad.view(
                        list(net.children())[0][0][0].weight.shape[0], -1).detach().cpu())
                elif type(net) == ResNet or type(net) == torchvision.models.resnet.ResNet:
                    weights.append(list(net.children())[0].weight.grad.view(list(net.children())[0].weight.shape[0],
                                                                            -1).detach().cpu())
            # entropy = Categorical(
            #    probs=torch.maximum(F.softmax(pred.detach().cpu(), dim=1), torch.tensor(1e-12)))  # F.softmax(pred.detach().cpu(), dim=1)
            # entropies += entropy.entropy().mean()
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
                        print("mean entropies:", entropies / (i + 1), file=file)
                        print(
                            f"Loss: {loss.detach().cpu()}, batch acc:{batchacc} progress: {int(((i + 1) / items) * 10000) / 100}%, Batch",
                            (i + 1), file=file)
                        print("Class accs:", class_accs[0] / (class_accs[1] + 1e-12), file=file)
                        print(int((time.time() - t0) * 100) / 100, "Seconds elapsed for 1 batch forward.",
                              "Average net call:", int((networkCallTime / (i + 1)) * 100000) / 100, "Milliseconds",
                              file=file)

                    if do_profiling:
                        stack.close()
                        print(prof.key_averages().table(row_limit=10), file=file)
                        with open("all_nets_profiling.txt", "a") as f:
                            print("network", run_name, ":", file=f)
                            print(prof.key_averages().table(row_limit=10), file=f)
                        return
                # l = 0
                accs = 0
        if make_imgs:  # todo histogram of gradients
            y, x = torch.histogram(torch.stack(weights))
            x = ((x + x.roll(-1))*0.5)[:-1]
            plt.bar(x, y, label="Gradient magnitude distribution", width=(x[-1]-x[0])/99)
            plt.xlabel("Bin limits")
            plt.yscale("log")
            plt.tight_layout()
            newpath = f"./imgs/{run_name}"
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            plt.show()
            plt.savefig(os.path.join(newpath, f"e{epoch}-b{i}.png"))
            # plt.show()
            plt.clf()
        # scheduler.step(l.item() / i)
        ep_losses.append(l.item() / (i + 1))
        losses.append(np.nan)
        if file is not None:
            print(f"Average Epoch {epoch} Train Loss:", l.item() / (i + 1), file=file)
            print(f"Epoch mean acc: {sum(train_accs) / (i + 1)}", file=file)
        print(f"Average Epoch {epoch} Train Loss:", l.item() / (i + 1))
        print("mean entropies:", entropies / (i + 1), file=file, end=" - ")
        print(f"Epoch mean acc: {sum(train_accs) / (i + 1)}")

def test(epoch, test_every_n, plot_loss, n_conv, device, loss_fn, test_losses, verbose, file, testitems,
         report_class_accs, ep_test_losses, eval_mode, net, dataset2, bs):
    test_accs = []
    if (epoch % test_every_n == (test_every_n - 1)) or plot_loss:
        net.eval()
        if eval_mode is not None:
            net._set_perforation([eval_mode] * n_conv)
        class_accs = np.zeros((2, 15))
        l2 = 0
        for i, (batch, classes) in enumerate(dataset2):

            pred = net(batch.to(device))
            loss = loss_fn(pred, classes.to(device))
            l2 += loss.detach().cpu()
            test_losses.append(loss.detach().cpu())
            acc = (F.softmax(pred.detach().cpu(), dim=1).argmax(dim=1) == classes)
            test_accs.append(torch.sum(acc) / bs)
            for clas in classes[acc]:
                class_accs[0, clas] += 1
            for clas in classes:
                class_accs[1, clas] += 1
            if i % 50 == 0 and verbose:
                print(
                    f"Loss: {loss.detach().cpu().item()}, batch acc:{acc.sum() / bs} progress: {int(((i + 1) / testitems) * 10000) / 100}%",
                    file=file)

                print("Class accs:", class_accs[0] / (class_accs[1] + 1e-12), file=file)
        test_losses.append(np.nan)
        if report_class_accs:
            print("Class accs:", class_accs[0] / (class_accs[1] + 1e-12), file=file)
        if file is not None:
            print("Average Epoch Test Loss:", l2.item() / (i + 1), file=file)
            print(f"Epoch mean acc: {sum(test_accs) / (i + 1)}", file=file)
        print("Average Epoch Test Loss:", l2.item() / (i + 1))
        print(f"Epoch mean acc: {sum(test_accs) / (i + 1)}")
        ep_test_losses.append(l2.item() / (i + 1))
def test_net(net, batch_size=128, verbose=False, epochs=10, summarise=False, run_name="", do_profiling=False,
             make_imgs=False, test_every_n=5, plot_loss=False, report_class_accs=False, vary_perf=None, eval_mode=None,
             file=None, dataset=None, dataset2=None, dataset3=None, op=None, lr_scheduler=None, validate=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bs = batch_size
    # transform = transforms.Compose(
    #    [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset = DataLoader(AppleDataset(mode="train"), num_workers=4, batch_size=bs, shuffle=True)
    if dataset is None:
        dataset = DataLoader(CINIC10(partition="train", download=True, transform=tf),  # collate_fn=col,
                             num_workers=4, batch_size=bs, shuffle=True, worker_init_fn=seed_worker,
                             generator=g)
    if dataset2 is None:
        dataset2 = DataLoader(
            CINIC10(partition="test", download=True, transform=tf), num_workers=4,  # collate_fn=col,
            batch_size=bs, shuffle=True, worker_init_fn=seed_worker,
            generator=g, )
    if dataset3 is None:
        dataset3 = DataLoader(
            CINIC10(partition="valid", download=True, transform=tf), num_workers=4,  # collate_fn=col,
            batch_size=bs, shuffle=True, worker_init_fn=seed_worker,
            generator=g, )
    # net = simpleNN(perf="both", interpolate_gradients=True)
    # net = Cifar10CnnModel(perf="both", interpolate_grad=True)
    loss_fn = nn.CrossEntropyLoss()
    if summarise:
        summary(net, input_size=(bs, 3, 32, 32))

    net.to(device)
    if op is None:
        op = optim.Adam(net.parameters(), lr=0.001)

    # scheduler = ReduceLROnPlateau(op, 'min')
    # op = optim.SGD(net.parameters(), lr=0.001)
    n_epochs = epochs
    items = len(dataset)
    testitems = len(dataset2)
    # TODO TRAIN TEST VALIDATE SPLIT
    best_accs = np.zeros(15)
    losses = []
    ep_losses = []
    test_losses = []
    ep_test_losses = []
    net.train()
    params = [1]
    n_conv = 0
    if hasattr(net, 'perforation'):
        n_conv = len(net.perforation)
    p = 0
    if eval_mode is not None:
        p = net._get_perforation()
    minacc = 10

    for epoch in range(n_epochs):
        train(do_profiling, dataset, n_conv, p, device, loss_fn, make_imgs, losses, op, verbose, file, items, epoch,
              ep_losses, vary_perf, eval_mode, net, bs, run_name)

        test(epoch, test_every_n, plot_loss, n_conv, device, loss_fn, test_losses, verbose, file, testitems,
         report_class_accs, ep_test_losses, eval_mode, net, dataset2, bs)

        if (epoch % test_every_n == (test_every_n - 1)) or plot_loss:
            if (ep_test_losses[-1]) < minacc:
                minacc = ep_test_losses[-1]
                params.pop()
                params.append(copy.deepcopy(net.state_dict()))
        net.train()
        if lr_scheduler is not None:
            lr_scheduler.step()
    if validate:
        net.load_state_dict(params[0])
        net.eval()
        if eval_mode is not None:
            net._set_perforation([eval_mode] * n_conv)
        class_accs3 = np.zeros((2, 15))
        l3 = 0
        valid_accs = []
        for ii, (batch, classes) in enumerate(dataset3):

            pred = net(batch.to(device))
            loss = loss_fn(pred, classes.to(device))
            l3 += loss.detach().cpu()
            acc = (F.softmax(pred.detach().cpu(), dim=1).argmax(dim=1) == classes)
            valid_accs.append(torch.sum(acc) / bs)
            for clas in classes[acc]:
                class_accs3[0, clas] += 1
            for clas in classes:
                class_accs3[1, clas] += 1
        print(f"Best test epoch:", file=file)
        print(f"Validation loss: {l3 / (ii + 1)}, validation class accs: {class_accs3[0] / (class_accs3[1] + 1e-12)}",
              file=file)

        print(f"Epoch mean acc: {sum(valid_accs) / (ii + 1)}", file=file)
        print(f"Validation mean acc: {sum(valid_accs) / (ii + 1)}")
    test_losses.append(np.nan)
    fig, axes = plt.subplots(1, 2, sharey=True,
                             figsize=(int(np.maximum(epochs, 15)), int(np.maximum(epochs // 1.5, 10))))
    axes[0].scatter(range(len(losses)), losses, label="losses", alpha=0.1)
    axes[1].scatter(range(len(test_losses) + (len(test_losses) // epochs)),
                    ([np.nan] * (len(test_losses) // epochs)) + test_losses, label="test losses", alpha=0.1)
    axes[0].plot(np.arange((len(losses) // epochs), len(losses) + (len(losses) // epochs), (len(losses) // epochs)),
                 ep_losses, color="r", label="Avg epoch Train loss")
    axes[1].plot(np.arange((len(losses) // epochs), len(losses) + (len(losses) // epochs), (len(losses) // epochs)),
                 ep_test_losses, color="yellow", label="Avg Epoch Test loss")
    # for ax in axes
    axes[0].set_xticks(np.arange(0, len(losses) + (len(losses) // epochs), (len(losses) // epochs)),
                       np.arange(0, epochs + 1, 1), rotation=90)
    axes[1].set_xticks(np.arange(0, len(losses) + (len(losses) // epochs), (len(losses) // epochs)),
                       np.arange(0, epochs + 1, 1), rotation=90)
    axes[0].set_xlabel("Epochs")
    axes[1].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(-0.15, 6)
    axes[1].set_ylim(-0.15, 6)
    axes[0].legend()
    axes[1].legend()
    axes[0].grid()
    axes[1].grid()
    plt.tight_layout()
    plt.savefig(f"./timelines/loss_timeline_{run_name}.png")
    # plt.show()
    plt.clf()


if __name__ == "__main__":
    # compare_speed()
    # quit()


    np.random.seed(0)
    random.seed(0)
    tf = transforms.Compose([transforms.ToTensor()])

    bs = 128
    dataset1 = DataLoader(CINIC10(partition="train", download=True, transform=tf),  # collate_fn=col,
                          num_workers=4, batch_size=bs, shuffle=True,
                          generator=g)
    dataset2 = DataLoader(
        CINIC10(partition="test", download=True, transform=tf), num_workers=4,  # collate_fn=col,
        batch_size=bs, shuffle=True,
        generator=g, )
    dataset3 = DataLoader(
        CINIC10(partition="valid", download=True, transform=tf), num_workers=4,  # collate_fn=col,
        batch_size=bs, shuffle=True,
        generator=g, )
    nets = [
        # TODO check the article for interpolation strategies
        #  todo different ways of interpolation/perforation (from the article) measuere the
        #   effect on just one layer(implement in architecture probably)
        #   work on gradient comparison on one layer thing but use the entire network so we can do the comparison
        #   while learning (just maybe do matrix difference images or something like that,
        #   such as gradient number size comparison in a distibution
        #   dont bother much (or at all) with speedup, focus on "accuracy" performance

        # TODO treniranje do saturationa lossa - da dobimo hitrost konvergence pri treniranju (run on something not my laptop)
        # TODO entropy of the output probability vector

    ]
    for n in [resnet18, MobileNetV2, mobilenet_v3_small, mobilenet_v3_large, resnet152]:
        for perf in ["none", "both", "trip", "start2"]:
            extra = ""
            if n == mobilenet_v3_small:
                extra += "small"
            elif n == mobilenet_v3_large:
                extra += "large"
            elif n == resnet18:
                extra += "18"
            elif n == resnet152:
                extra += "152"
            if perf == "start2":
                if n == mobilenet_v3_small:
                    perf = ["both"] + ["none"] * 51
                    extra += "-"
                elif n == MobileNetV2:
                    perf = ["both"] + ["none"] * 51
                elif n == mobilenet_v3_large:
                    perf = ["both"] + ["none"] * 61
                    extra += "-"
                elif n == resnet18:
                    perf = ["both"] + ["none"] * 24
                    extra += "-"
                elif n == resnet152:
                    extra += "-"
                    perf = ["both"] + ["none"] * 200
                extra += "only_1st_perf"
            for grad in [True, False]:
                nets.append(n(num_classes=10, perforation_mode=perf, grad_conv=grad, extra_name=extra))
    i = 0
    if not os.path.exists("./results"):
        os.mkdir("./results")
    for net in nets:
        for eval_mode in ["none", "both", "trip"]:
            for vary_perf in [None]:  # , "random"]:  # , "incremental"]:
                # TODO SEPARATE CODE INTO SEPARATE FUNCTIONS THIS IS UGLY AF
                # TODO run convergence tests on fri machine
                # vary_perf = "random"
                run_name = type(net).__name__ + "-" + \
                           (net.extra_name + "-" if net.extra_name != "" else "") + \
                           "perf_" + (f"{vary_perf}" if vary_perf is not None else net.perforation[0]) + \
                           "-eval_" + eval_mode + \
                           f"-grad_{net.grad_conv}"
                i += 1
                plot_loss = False
                validate=False
                run_name += "_short"
                if plot_loss:
                    if os.path.exists(f"./timelines/loss_timeline_{run_name}.png") and \
                            os.path.exists(f"./results/results_{run_name}.png"):
                        print(f"Run {run_name} already complete, skipping...")
                        continue
                    else:
                        print(f"Starting {run_name}...")
                # print(run_name)
                with open(f"./results/results_{run_name}.txt", "w") as f:
                    t = time.time()

                    test_net(net, batch_size=bs, epochs=20, do_profiling=False, summarise=False, verbose=False,
                             make_imgs=True, plot_loss=plot_loss, vary_perf=vary_perf, file=f, eval_mode=eval_mode,
                             run_name=run_name, dataset=dataset1, dataset2=dataset2, dataset3=dataset3, validate=validate)
                    duration = time.time() - t
                    print(f"{run_name}\n{duration} seconds Elapsed", file=f)
                    print(f"{run_name}\n{duration} seconds Elapsed")

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
