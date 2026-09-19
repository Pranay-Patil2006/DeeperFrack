import matplotlib.pyplot as plt
import numpy as np

# Data
architectures = ['VGG02', 'AlexNet', 'MobileNet']
methods = ['Naive', 'DeepFrack\n(+ Precompute)', 'PolyFuse']

# JSON values in uJ, converted to mJ to match Figure 9 of the paper
energy_data = {
    'VGG02': [36449.1 / 1000, 10409.6 / 1000, 10491.5 / 1000],
    'AlexNet': [2908.8 / 1000, 2722.6 / 1000, 1635.4 / 1000],
    'MobileNet': [2558.2 / 1000, 729.4 / 1000, 729.4 / 1000]
}

# DeepFrack times from paper: ~6 hours benchmark + search time
time_data = {
    'VGG02': ['N/A', '~19 hours\n(Invalid)', '0.05s'],
    'AlexNet': ['N/A', '~6 hours\n(Invalid)', '0.04s'],
    'MobileNet': ['N/A', '~6 hours', '0.06s']
}

colors = ['#e74c3c', '#f39c12', '#2ecc71']
hatches = ['', '//', '']

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=False)
fig.suptitle('Energy Reduction & Total Wall-Clock Time: Exact Paper Values', fontsize=16)

for ax, arch in zip(axes, architectures):
    x = np.arange(len(methods))
    bars = ax.bar(x, energy_data[arch], color=colors, edgecolor='black')
    
    for i, bar in enumerate(bars):
        bar.set_hatch(hatches[i])
        
    ax.set_title(f"{arch} (Simba)", fontsize=14)
    ax.set_ylabel('Energy (mJ)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, max(energy_data[arch]) * 1.2)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        time_str = time_data[arch][i]
        text = f"{height:.1f} mJ\nTime: {time_str}"
        ax.annotate(text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

fig.tight_layout()
plt.savefig('/deeper/src/subgraphs_gains.png', dpi=300)
print("Graph saved!")
