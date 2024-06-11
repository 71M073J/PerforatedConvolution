import matplotlib.pyplot as plt

ks = [3,5, 7]
fig, axes = plt.subplots(1,len(ks), sharey=True, figsize=(5 * len(ks),5))
for i, k in enumerate(ks):
    norm = []
    x2 = []
    x3 = []
    for n in range(2, 2000):
        norm.append(k**2 * n**2)
        perf = 2
        x2.append((n//perf)**2 * k ** 2 + (perf**2-1)/perf**2 * n * n)
        perf = 3
        x3.append((n//perf)**2 * k ** 2 + (perf**2-1)/perf**2 * n * n * 2)
    #axes[i].set_title("")
    axes[i].plot(norm, label=f"Normal conv, kernel={k}, N_ops: {1}")
    axes[i].plot(x2, label=f"2x2 perf conv, kernel={k}, N_ops: {int((x2[-1]/norm[-1]) * 1000)/1000}")
    axes[i].plot(x3, label=f"3x3 perf conv, kernel={k}, N_ops: {int((x3[-1]/norm[-1]) * 1000)/1000}")
    axes[i].legend()
axes[0].set_ylabel("N_ops")
axes[0].set_xlabel("Input size")
axes[1].set_xlabel("Input size")
axes[2].set_xlabel("Input size")
plt.savefig("./efficiency_curves.png")
plt.show()