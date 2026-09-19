import sys
import matplotlib.pyplot as plt
import numpy as np

def generate_plot(polyfuse_energy):
    naive_energy = 2908.83
    deepfrack_energy = 2722.65 # DeepFrack's actual output due to Cheatsheet constraints
    polyfuse_energy = float(polyfuse_energy)

    labels = ['Naive', 'DeepFrack', 'PolyFuse']
    energies = [naive_energy, deepfrack_energy, polyfuse_energy]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    plt.figure(figsize=(10, 6), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#121212')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.tick_params(axis='x', colors='white', labelsize=12)
    ax.tick_params(axis='y', colors='white', labelsize=12)

    bars = plt.bar(labels, energies, color=colors, width=0.6)
    
    plt.title('AlexNet Energy Consumption (Simba Native Timeloop)', color='white', pad=20, fontsize=16, fontweight='bold')
    plt.ylabel('Energy (\u03bcJ)', color='white', fontsize=14, fontweight='bold')
    plt.ylim(0, max(energies) * 1.2)

    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                 f'{height:,.1f} \u03bcJ',
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=11)

    # Calculate gains relative to naive
    df_gain = (naive_energy - deepfrack_energy) / naive_energy * 100
    pf_gain = (naive_energy - polyfuse_energy) / naive_energy * 100
    
    # Add floating text for gains
    plt.text(1, deepfrack_energy/2, f'(-{df_gain:.1f}%)', ha='center', va='center', color='#121212', fontweight='bold', fontsize=14)
    plt.text(2, polyfuse_energy/2, f'(-{pf_gain:.1f}%)', ha='center', va='center', color='#121212', fontweight='bold', fontsize=14)

    plt.grid(axis='y', color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('/deeper/src/alexnet_pure_comparison.png', dpi=300, facecolor='#121212', edgecolor='none')
    print("Plot saved to /deeper/src/alexnet_pure_comparison.png")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_plot(sys.argv[1])
    else:
        print("Please provide the polyfuse energy as an argument.")
