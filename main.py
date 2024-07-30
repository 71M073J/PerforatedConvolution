import os.path
import time
import copy
from typing import Union

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
    # device = "cpu"
    timing = "forward"
    # device = "cpu"
    in_channels = 64
    backwards = True
    for kern in [1, 3, 5]:
        for in_channels in [2, 64, 256]:
            # l1 = nn.Conv2d(in_channels, in_channels * 2, 5).to(device)
            l1 = PerforatedConv2d(in_channels, in_channels * 2, kern, perforation_mode=(1, 1)).to(device)
            l2 = PerforatedConv2d(in_channels, in_channels * 2, kern, perforation_mode=(2, 2)).to(device)
            l3 = PerforatedConv2d(in_channels, in_channels * 2, kern, perforation_mode=(3, 3)).to(device)
            l4 = PerforatedConv2d(in_channels, in_channels * 2, kern, perforation_mode=(4, 4)).to(device)
            l5 = PerforatedConv2d(in_channels, in_channels * 2, kern, perforation_mode=(5, 5)).to(device)
            ls = [l1, l2, l3, l4, l5]
            os = [torch.optim.SGD(l.parameters(), lr=.001) for l in ls]
            total_times = [0, 0, 0, 0, 0]
            times = [[], [], [], [], []]
            # with torch.no_grad():
            iters = 10
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
            for i in range(len(ls)):
                print(
                    f"Perforated convLayer: {ls[i].perf_stride}: average of {total_times[i] / len(times[i])} seconds")
            for i in range(len(ls)):
                plt.plot(times[i],
                         label=f"PerfConv {ls[i].perf_stride}, {int(1000 * (total_times[i] / len(times[i]) * 1000)) / 1000}ms avg")
            plt.legend()
            plt.ylabel("Time elapsed (s)")
            if backwards:
                plt.savefig(f"{device[:4]}_kern{kern}_in_channels_{in_channels}_with_backward.png")
            else:
                plt.savefig(f"{device[:4]}_in_channels_{in_channels}.png")
            plt.show()
            plt.clf()


def train(do_profiling, dataset, n_conv, p, device, loss_fn, make_imgs, losses, op, verbose, file, items, epoch,
          ep_losses, vary_perf, eval_mode, net, bs, run_name, reporting, vary_num):
    l = 0
    train_accs = []
    entropies = 0
    networkCallTime = 0
    class_accs = np.zeros((2, 15))
    weights = []
    net._set_perforation(p)
    with ExitStack() as stack:
        if do_profiling:
            prof = stack.enter_context(torch.autograd.profiler.profile(profile_memory=True, use_cuda=True))
        for i, (batch, classes) in enumerate(dataset):
            batchacc = 0
            if vary_perf is not None:
                # raise NotImplementedError("TODO with tuples")
                perfs = None
                if vary_perf == "incremental":
                    raise NotImplementedError()

                elif vary_perf == "random":#avg_proc 0.37 of non-perf
                    #rn = np.random.random(n_conv)
                    #perfs = np.array([(1, 1)] * n_conv)
                    perfs = np.random.randint(1, vary_num+1, (n_conv,2))
                    #for i in range(1, vary_num):
                    #    pmask = rn > 1. / vary_num
                    #    if sum(pmask) != 0:
                    #        perfs[pmask] = np.array([(i + 1, i + 1)] * len(perfs[pmask]))
                elif vary_perf == "2by2_equivalent":
                    perfs = np.ones((n_conv, 2), dtype=int)
                    rns = np.random.random(n_conv)
                    for i in range(n_conv):
                        rn = rns[i]
                        if rn < 0.20029760897159576:
                            perfs[i] = 3,3
                        elif rn < 0.32178425043821335:
                            perfs[i] = 2,3
                        elif rn < 0.44327089190483093:
                            perfs[i] = 3,2
                        elif rn < 0.5378847792744637:
                            perfs[i] = 3,1
                        elif rn < 0.6407210007309914:
                            perfs[i] = 1,3
                        elif rn < 0.7435572221875191:
                            perfs[i] = 2,2
                        elif rn < 0.8306062072515488:
                            perfs[i] = 1,2
                        elif rn < 0.9176551923155785:
                            perfs[i] = 2,1
                        else:
                            perfs[i] = 1,1
                else:
                    raise ValueError("Only supported vary modes are random and 2by2_equivalent")
                net._set_perforation(perfs)
            batch = batch.to(device)
            t0 = time.time()
            pred = net(batch)
            acc = F.softmax(pred.detach().cpu(), dim=1).argmax(dim=1) == classes
            train_accs.append(torch.sum(acc) / bs)
            torch.cuda.synchronize()
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
            # acc = (F.softmax(pred.detach().cpu(), dim=1).argmax(dim=1) == classes)
            for clas in classes[acc]:
                class_accs[0, clas] += 1
            for clas in classes:
                class_accs[1, clas] += 1
            l += loss.detach().cpu()
            losses.append(loss.detach().cpu())
            op.step()
            op.zero_grad()
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
    if make_imgs:  # todo histogram of gradients
        y, x = torch.histogram(torch.stack(weights))
        x = ((x + x.roll(-1)) * 0.5)[:-1]
        plt.bar(x, y, label="Gradient magnitude distribution", width=(x[-1] - x[0]) / 99)
        plt.xlabel("Bin limits")
        plt.yscale("log")
        plt.tight_layout()
        newpath = f"./imgs/{run_name}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        plt.savefig(os.path.join(newpath, f"grad_hist_e{epoch}.png"))
        # plt.show()
        plt.clf()
    # scheduler.step(l.item() / i)
    ep_losses.append(np.mean(losses).item())
    # losses.append(np.nan)
    if reporting:
        if file is not None:
            print(f"Average Epoch {epoch} Train Loss:", np.mean(losses).item(), file=file)
            print(f"Epoch mean acc: {np.mean(train_accs).item()}", file=file)
        print(f"Average Epoch {epoch} Train Loss:", np.mean(losses).item())
        # print("mean entropies:", entropies / (i + 1), file=file, end=" - ")
        print(f"Epoch mean acc: {np.mean(train_accs).item()}")


def test(epoch, test_every_n, plot_loss, n_conv, device, loss_fn, test_losses, verbose, file, testitems,
         report_class_accs, ep_test_losses, eval_mode, net, dataset2, bs, reporting, test_accs):
    train_mode = ""
    if hasattr(net, "perforation"):
        train_mode = net._get_perforation()
    with torch.no_grad():
        if (epoch % test_every_n == (test_every_n - 1)) or plot_loss:
            net.eval()
            if eval_mode is not None:
                if type(eval_mode) == tuple:
                    net._set_perforation([eval_mode] * n_conv)
                else:
                    net._set_perforation(eval_mode)
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
            if reporting:
                if report_class_accs:
                    print("Class accs:", class_accs[0] / (class_accs[1] + 1e-12), file=file)
                if file is not None:
                    print("Average Epoch Test Loss:", l2.item() / (i + 1), file=file)
                    print(f"Epoch mean acc: {np.mean(test_accs).item()}", file=file)
                print("Average Epoch Test Loss:", l2.item() / (i + 1))
                print(f"Epoch mean acc: {np.mean(test_accs).item()}")
            ep_test_losses.append(l2.item() / (i + 1))

    if hasattr(net, "perforation"):
        net._set_perforation(train_mode)


def test_net(net, batch_size=128, verbose=False, epochs=10, summarise=False, run_name="", do_profiling=False,
             make_imgs=False, test_every_n=1, plot_loss=False, report_class_accs=False, vary_perf=None, eval_mode=None,
             file=None, dataset=None, dataset2=None, dataset3=None, op=None, lr_scheduler=None, validate=None,
             do_test=True, save_net=False, reporting=True, vary_num=2, device: Union[str, torch.device] = "cpu"):
    if vary_perf is not None:
        try:
            vary_num = net.perforation[0][0]
        except:
            raise ValueError("net perf incompatible with vary num")
    bs = batch_size
    if validate is None:
        if dataset3 is None:
            validate = False
        else:
            validate = True
    if do_test and test_every_n > epochs:
        do_test = False
    if op is None:
        op = torch.optim.Adam(net.parameters(), lr=0.001)
    # net = simpleNN(perf="both", interpolate_gradients=True)
    # net = Cifar10CnnModel(perf="both", interpolate_grad=True)
    loss_fn = nn.CrossEntropyLoss()
    if summarise:
        summary(net, input_size=(bs, 3, 32, 32))

    if dataset3 is None:
        dataset3 = dataset2
    net.to(device)

    # scheduler = ReduceLROnPlateau(op, 'min')
    # op = optim.SGD(net.parameters(), lr=0.001)
    n_epochs = epochs
    items = len(dataset)
    testitems = len(dataset2)
    losses = []
    ep_losses = []
    test_losses = []
    ep_test_losses = []
    best_acc = 0
    net.train()
    params = [1]
    n_conv = 0
    if hasattr(net, 'perforation'):
        n_conv = len(net.perforation)
    p = net._get_perforation()
    minacc = 1000
    for epoch in range(n_epochs):
        train(do_profiling, dataset, n_conv, p, device, loss_fn, make_imgs, losses, op, verbose, file, items, epoch,
              ep_losses, vary_perf, eval_mode, net, bs, run_name, reporting, vary_num)

        if type(eval_mode) == tuple:
            if do_test or plot_loss:
                test_accs = []
                test(epoch, test_every_n, plot_loss, n_conv, device, loss_fn, test_losses, verbose, file, testitems,
                     report_class_accs, ep_test_losses, eval_mode, net, dataset2, bs, reporting, test_accs)

            if lr_scheduler is not None:
                lr_scheduler.step()
            if (ep_test_losses[-1]) < minacc:
                minacc = ep_test_losses[-1]
                best_acc = np.mean(test_accs)
                params.pop()
                params.append(copy.deepcopy(net.state_dict()))
        elif type(eval_mode) == list:
            if do_test or plot_loss:
                test_accs = []
                test_accs = [test_accs] * len(eval_mode)
                test_losses = [test_losses] * len(eval_mode)
                ep_test_losses = [ep_test_losses] * len(eval_mode)
                params = [params] * len(eval_mode)
                best_acc = [0] * len(eval_mode)
                if type(minacc) != list:
                    minacc = [1000.] * len(eval_mode)
                for ind, ev in enumerate(eval_mode):

                    print("testing eval mode", ev)
                    print("testing eval mode", ev, file=file)
                    test(epoch, test_every_n, plot_loss, n_conv, device, loss_fn, test_losses[ind], verbose, file,
                         testitems, report_class_accs, ep_test_losses[ind], ev, net, dataset2, bs, reporting,
                         test_accs[ind])

                    if (ep_test_losses[ind][-1]) < minacc[ind]:
                        minacc[ind] = ep_test_losses[ind][-1]
                        best_acc[ind] = np.mean(test_accs[ind])
                        params[ind].pop()
                        params[ind].append(copy.deepcopy(net.state_dict()))
                if lr_scheduler is not None:
                    lr_scheduler.step()
        net.train()
    if validate:
        net.load_state_dict(params[0])
        net.eval()
        if eval_mode is not None:
            net._set_perforation([eval_mode] * n_conv)
        class_accs3 = np.zeros((2, 15))
        l3 = 0
        valid_accs = []
        with torch.no_grad():
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
    if save_net:
        torch.save(params, f"{run_name}_model.pth")
    if plot_loss:
        # test_losses.append(np.nan)
        fig, axes = plt.subplots(1, 2 if do_test else 1, sharey=True,
                                 figsize=(int(np.maximum(epochs, 15)), int(np.maximum(epochs // 1.5, 10))))
        ax1 = axes[0] if do_test else axes
        ax1.scatter(range(len(losses)), losses, label="losses", alpha=0.1)
        ax1.plot(np.arange((len(losses) // epochs), len(losses) + (len(losses) // epochs), (len(losses) // epochs)),
                 ep_losses, color="r", label="Avg epoch Train loss")

        # for ax in axes
        ax1.set_xticks(np.arange(0, len(losses) + (len(losses) // epochs), (len(losses) // epochs)),
                       np.arange(0, epochs + 1, 1), rotation=90)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        if do_test:
            test_losses = np.array(test_losses)
            axes[0].errorbar(
                np.arange((len(losses) // epochs), len(losses) + (len(losses) // epochs), (len(losses) // epochs)),
                ep_test_losses,
                [test_losses[x * (len(test_losses) // epochs):(x + 1) * (len(test_losses) // epochs - 1)].std() ** 0.5
                 for x in range(epochs)], capsize=3,
                label="Test losses")

        ax1.set_ylim(-0.15, 8)
        ax1.legend()
        ax1.grid()
        plt.tight_layout()
        plt.savefig(f"./timelines/loss_timeline_{run_name}.png")
        # plt.show()
        plt.clf()
    return best_acc


def compare_speed_net():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    augment = True
    tf = [transforms.ToTensor(), ]
    tf_test = [transforms.ToTensor(), ]
    dataset1, dataset2, dataset3 = None, None, None
    bs = 64
    if augment:
        tf.extend([transforms.RandomChoice([transforms.RandomCrop(size=32, padding=4),
                                            transforms.RandomResizedCrop(size=32)]),
                   transforms.RandomHorizontalFlip()])
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
    cnt = 0
    for n in [MobileNetV2, mobilenet_v3_small, resnet18]:  # , mobilenet_v3_large, resnet152]:
        for perf in [(2, 2), (3, 3), (1, 1)]:  # , "largest"
            for grad in [True]:  # , False]:
                net = n(num_classes=10, perforation_mode=perf, grad_conv=grad)
                cnt += 1
                op = torch.optim.Adam(net.parameters(), weight_decay=0.001)
                t = time.time()
                test_net(net, batch_size=bs, epochs=1, do_profiling=False, summarise=False, verbose=False,
                         make_imgs=False, plot_loss=False, vary_perf=None, file=None, eval_mode=(2, 2),
                         run_name=f"benchmarking{cnt}", dataset=dataset1, dataset2=dataset2, dataset3=None,
                         validate=False, test_every_n=1, op=op, lr_scheduler=None, device=device)
                print(f"net {n} {perf} took {time.time() - t} seconds.")


def do_profiling():
    augment = True
    tf = [transforms.ToTensor(), ]
    tf_test = [transforms.ToTensor(), ]
    data = "cifar"
    dataset1, dataset2, dataset3 = None, None, None
    g = None
    bs = 128
    if data == "cinic":
        tf.append(transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                       [0.24205776, 0.23828046, 0.25874835]))
        tf_test.append(transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                            [0.24205776, 0.23828046, 0.25874835]))
        tf = transforms.Compose(tf)
        tf_test = transforms.Compose(tf_test)
        dataset1 = DataLoader(CINIC10(partition="train", download=True, transform=tf),  # collate_fn=col,
                              num_workers=4, batch_size=bs, shuffle=False,
                              generator=g)
        dataset2 = DataLoader(
            CINIC10(partition="valid", download=True, transform=tf_test), num_workers=4,  # collate_fn=col,
            batch_size=bs, shuffle=False,
            generator=g, )
        dataset3 = DataLoader(
            CINIC10(partition="test", download=True, transform=tf_test), num_workers=4,  # collate_fn=col,
            batch_size=bs, shuffle=False,
            generator=g, )
    elif data == "cifar":
        tf.extend([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        tf_test.extend([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        tf = transforms.Compose(tf)
        tf_test = transforms.Compose(tf_test)
        dataset1 = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=tf), batch_size=bs, shuffle=False, num_workers=4)

        dataset2 = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=tf_test), batch_size=bs, shuffle=False,
            num_workers=4)
    for device in ["cpu", "cuda:0"]:
        for perf in [(2,2), (1,1)]:
            with open("./all_nets_profiling.txt", "a") as ggsdf:
                for i in range(10):
                    print("-"*50, "\n", "-", device, " and ",perf, file=ggsdf)
            net = resnet18(num_classes=10, perforation_mode=perf)
            test_net(net, do_profiling=True, epochs=4, dataset=dataset1, dataset2=dataset2, device=device)



if __name__ == "__main__":
    #do_profiling()
    #quit()
    # compare_speed()
    # quit()
    # compare_speed_net()
    # quit()
    print(
        "TODOTODOTODO important! naredi verzijo kjer namesto da se gradient downscalea, se direktno preko konvolucije pošlje?")
    augment = True
    tf = [transforms.ToTensor(), ]
    tf_test = [transforms.ToTensor(), ]
    data = "cifar"
    dataset1, dataset2, dataset3 = None, None, None
    g = None
    bs = 128

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if augment:
        tf.extend([transforms.RandomChoice([transforms.RandomCrop(size=32, padding=4),
                                            transforms.RandomResizedCrop(size=32)]),
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

        # TODO check the article for interpolation strategies
        #  todo different ways of interpolation/perforation (from the article) measuere the
        #   effect on just one layer(implement in architecture probably)
        #   work on gradient comparison on one layer thing but use the entire network so we can do the comparison
        #   while learning (just maybe do matrix difference images or something like that,
        #   such as gradient number size comparison in a distibution
        #   dont bother much (or at all) with speedup, focus on "accuracy" performance

        #
        # TODO entropy of the output probability vector - ugotovi zakaj ne dela prav

    i = 0
    if not os.path.exists("./results"):
        os.mkdir("./results")
    # nets = [resnet18(num_classes=10, perforation_mode=(2, 2), grad_conv=True)]
    for n in [resnet18, MobileNetV2, mobilenet_v3_small]:  # , mobilenet_v3_large, resnet152]:
        for perf in [(2, 2), (3, 3), (1, 1)]:  # , "largest"
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
                    perf = [(2, 2)] + [(1, 1)] * 51
                    extra += "-"
                elif n == MobileNetV2:
                    perf = [(2, 2)] + [(1, 1)] * 51
                elif n == mobilenet_v3_large:
                    perf = [(2, 2)] + [(1, 1)] * 61
                    extra += "-"
                elif n == resnet18:
                    perf = [(2, 2)] + [(1, 1)] * 19
                    extra += "-"
                elif n == resnet152:
                    extra += "-"
                    perf = [(2, 2)] + [(1, 1)] * 200
                extra += "only_1st_perf"
            for grad in [True]:  # , False]:
                net = n(num_classes=10, perforation_mode=perf, grad_conv=grad, extra_name=extra)
                for eval_mode in [(1, 1)]:  # , (2, 2), (3, 3)]:
                    for vary_perf in [None]:  # , "random"]:  # , "incremental"]:
                        # TODO SEPARATE CODE INTO SEPARATE FUNCTIONS THIS IS UGLY AF
                        # TODO run convergence tests on fri machine
                        # vary_perf = "random"
                        run_name = type(net).__name__ + "-" + \
                                   (net.extra_name + "-" if net.extra_name != "" else "") + \
                                   "perf_" + (
                                       f"{vary_perf}" if vary_perf is not None else f"{net.perforation[0][0]}_{net.perforation[0][1]}") + \
                                   "-eval_" + f"{eval_mode[0]}_{eval_mode[1]}" + \
                                   f"-grad_{net.grad_conv}"
                        i += 1
                        plot_loss = False
                        validate = True
                        test_every_n = 1
                        # run_name += "_short"
                        if plot_loss:
                            if os.path.exists(f"./timelines/loss_timeline_{run_name}.png") and \
                                    os.path.exists(f"./results/results_{run_name}.txt"):
                                print(f"Run {run_name} already complete, skipping...")
                                continue
                            else:
                                print(f"Starting {run_name}...")
                        # print(run_name)
                        make_imgs = True
                        if make_imgs:
                            if os.path.exists(f"./imgs/{run_name}/grad_hist_e19.png") and \
                                    os.path.exists(f"./results/results_{run_name}.txt"):
                                print(f"Run {run_name} already complete, skipping...")
                                continue
                            else:
                                print(f"Starting {run_name}...")
                        with open(f"./results/results_{run_name}.txt", "w") as f:
                            t = time.time()
                            print(run_name)
                            op = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0., )
                            # op = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0005, )
                            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=200)
                            lr_scheduler = None
                            test_net(net, batch_size=bs, epochs=2, do_profiling=False, summarise=False, verbose=False,
                                     make_imgs=make_imgs, plot_loss=plot_loss, vary_perf=vary_perf, file=f,
                                     eval_mode=eval_mode,
                                     run_name=run_name, dataset=dataset1, dataset2=dataset2, dataset3=dataset3,
                                     validate=validate, test_every_n=test_every_n, op=op, lr_scheduler=lr_scheduler,
                                     device=device)
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
