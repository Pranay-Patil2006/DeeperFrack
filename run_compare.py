import sys
import os
import json
from polyfuse.timeloop_integration import TimeloopIntegration
from polyfuse.hal import NetworkLoader, HAL

def run_comparison(arch, components, network, name, df_part):
    print(f"\n=====================================")
    print(f" Starting Map-Space Evaluation: {name}")
    print(f"=====================================")
    
    loader = NetworkLoader(network)
    layers = loader.load()
    if not layers:
        print("No layers found!")
        return

    # PolyFuse DP Partition Simulation
    # For now, assume PolyFuse perfectly fuses into 1 block (as it often does for max energy savings if global buffer allows)
    pf_part = [[i for i in range(len(layers))]]
    
    # Naive Partition
    naive_part = [[i] for i in range(len(layers))]

    tl = TimeloopIntegration(arch, components, f'output/comparison_{name.replace(" ", "_")}')
    
    print("Evaluating Naive Layer-by-Layer mapping...")
    naive_e = 0
    for stack_idx, indices in enumerate(naive_part):
        stack_layers = [layers[i] for i in indices]
        res = tl.run_timeloop(stack_idx, stack_layers, {l['name']: {'T_out': l['P']} for l in stack_layers})
        naive_e += res['energy_pJ']
        
    print("Evaluating DeepFrack mapping...")
    df_e = 0
    tl = TimeloopIntegration(arch, components, f'output/comparison_{name.replace(" ", "_")}_df')
    for stack_idx, indices in enumerate(df_part):
        stack_layers = [layers[i] for i in indices]
        res = tl.run_timeloop(stack_idx, stack_layers, {l['name']: {'T_out': l['P']} for l in stack_layers})
        df_e += res['energy_pJ']

    print("Evaluating PolyFuse Optimal mapping...")
    pf_e = 0
    tl = TimeloopIntegration(arch, components, f'output/comparison_{name.replace(" ", "_")}_pf')
    for stack_idx, indices in enumerate(pf_part):
        stack_layers = [layers[i] for i in indices]
        res = tl.run_timeloop(stack_idx, stack_layers, {l['name']: {'T_out': l['P']} for l in stack_layers})
        pf_e += res['energy_pJ']
        
    print(f"\nResults for {name}:")
    print(f"  Naive implementation: {naive_e / 1e6:.2f} mJ")
    print(f"  DeepFrack Optimizer:  {df_e / 1e6:.2f} mJ")
    print(f"  PolyFuse Optimizer:   {pf_e / 1e6:.2f} mJ")

if __name__ == '__main__':
    # AlexNet
    run_comparison(
        '/app/Examples/AlexNet_Simba/simba_like/arch/simba_like.yaml',
        'data/components/',
        '/app/Examples/AlexNet_Simba/AlexNet/',
        'AlexNet',
        [[0, 1, 2, 3], [4]] # DeepFrack partition for AlexNet
    )
