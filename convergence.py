import os
import matplotlib.pyplot as plt
for train in [1,2,3]:
    for eval in [2]:#[1,2,3]:
        eval = train
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        with open(f"./all/resnet_perf_long_test{train}x{train}_out_eval{eval}x{eval}.txt", "r") as f:
            for i, line in enumerate(f):
                if i % 4 == 0:
                    train_losses.append(float(line.split()[5]))
                elif i % 4 == 1:
                    train_accs.append(float(line.split()[3]))
                elif i % 4 == 2:
                    test_losses.append(float(line.split()[8]))
                elif i % 4 == 3:
                    test_accs.append(float(line.split()[3]))
    plt.plot(test_losses, label=f"train/eval perf: {train}")

with open(f"./all/resnet18_large_starting_lr.txt", "r") as f:
    train_losses_lr = []
    test_losses_lr = []
    train_accs_lr = []
    test_accs_lr = []
    i = 0
    for i, line in enumerate(f):
        if i % 14 == 0:
            train_losses_lr.append(float(line.split()[5]))
        elif i % 14 == 1:
            train_accs_lr.append(float(line.split()[3]))
        elif i % 14 == 9:
            test_losses_lr.append(float(line.split()[4]))
        elif i % 14 == 10:
            test_accs_lr.append(float(line.split()[3]))

plt.plot(test_losses_lr, label="train/eval_small_start_lr")
plt.legend()
plt.show()