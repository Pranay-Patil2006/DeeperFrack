import matplotlib.pyplot as plt

methods = ['Naive (No Fusion)', 'DeepFrack (Bugged)', 'Analytical Optimizer']
energy = [2908.83, 2722.65, 1635.41]
reduction = [0, 6.40, 43.78]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, energy, color=['red', 'orange', 'green'])
plt.ylabel('Energy (uJ)')
plt.title('AlexNet on Simba: Energy Comparison')

for bar, red in zip(bars, reduction):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 50, f"{yval:.2f} uJ\n({red}% reduction)", ha='center', va='bottom', fontweight='bold')

plt.ylim(0, 3500)
plt.savefig('/deeper/src/alexnet_gains.png')
print("Graph saved!")
