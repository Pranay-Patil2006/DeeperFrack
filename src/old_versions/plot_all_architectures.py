import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['VGG02 (Simba)', 'AlexNet (Simba)', 'MobileNet (Simba)']
naive = [36449.1, 2908.8, 2558.2]
deepfrack = [10409.6, 2722.6, 729.4]
analytical = [10409.6, 1635.4, 729.4]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - width, naive, width, label='Naive (No Fusion)', color='#e74c3c')
rects2 = ax.bar(x, deepfrack, width, label='DeepFrack (Iterative Search)', color='#f39c12', hatch='//')
rects3 = ax.bar(x + width, analytical, width, label='PolyFuse Analytical', color='#2ecc71')

ax.set_ylabel('Energy (uJ)')
ax.set_title('Energy Reduction Across Architectures: Analytical vs Simulation Search')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects, annotations=None):
    for idx, rect in enumerate(rects):
        height = rect.get_height()
        text = f"{height:.1f}"
        if annotations and annotations[idx]:
            text += f"\n{annotations[idx]}"
        ax.annotate(text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2, annotations=["(Invalid)", "(Invalid)", ""])
autolabel(rects3)

fig.tight_layout()
plt.savefig('/deeper/src/all_architectures_gains.png', dpi=300)
print("Graph saved to /deeper/src/all_architectures_gains.png")
