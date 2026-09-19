import os
import subprocess
import yaml
import re
import matplotlib.pyplot as plt

def run_timeloop(problem_yaml, name):
    print(f"Running Timeloop for {name}...")
    arch_yaml = "/app/Examples/VGG_Simba/simba_like/arch/simba_like.yaml"
    comp_dir = "/app/Examples/VGG_Simba/simba_like/arch/components"
    components = [os.path.join(comp_dir, f) for f in os.listdir(comp_dir) if f.endswith('.yaml')]
    mapper_yaml = "/app/Examples/VGG_Simba/simba_like/mapper/mapper.yaml"
    
    constraints_yaml = "/app/Examples/VGG_Simba/simba_like/constraints/SLC.yaml"
    cmd = ["/opt/timeloop/bin/timeloop-mapper", arch_yaml] + components + [mapper_yaml, constraints_yaml, problem_yaml]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/timeloop/lib:/opt/timeloop/build:" + env.get("LD_LIBRARY_PATH", "")
    
    # Run in a dedicated output directory to avoid clashing stats files
    out_dir = f"/deeper/src/out_{name}"
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        result = subprocess.run(cmd, env=env, cwd=out_dir, capture_output=True, text=True, timeout=120)
        
        # Parse the stats.txt file
        stats_file = os.path.join(out_dir, "timeloop-mapper.stats.txt")
        if not os.path.exists(stats_file):
            print(f"Timeloop failed for {name}. Output:")
            print(result.stdout)
            print(result.stderr)
            return None, None
            
        with open(stats_file, 'r') as f:
            content = f.read()
            
        energy_match = re.search(r'Energy\s*:\s*([0-9.]+)\s*uJ', content)
        cycles_match = re.search(r'Cycles\s*:\s*([0-9]+)', content)
        
        energy = float(energy_match.group(1)) if energy_match else 0
        cycles = int(cycles_match.group(1)) if cycles_match else 0
        
        print(f"  -> Energy: {energy} uJ, Cycles: {cycles}")
        return energy, cycles
    except subprocess.TimeoutExpired:
        print(f"Timeloop timed out for {name}!")
        return None, None

def generate_virtual_prob(base_prob_path, out_path, new_P, new_Q):
    with open(base_prob_path, 'r') as f:
        data = yaml.safe_load(f)
        
    data['problem']['instance']['P'] = new_P
    data['problem']['instance']['Q'] = new_Q
    
    with open(out_path, 'w') as f:
        yaml.dump(data, f)
    return out_path

def main():
    # Phase 1: Mathematical sizing for VGG02 Layer 02
    # Layer 02 config: R=3, S=3, M=64, C=64, P=224, Q=224
    # Our analytical tiler calculated optimal tiles: T_out=32, T_in=34
    T_out = 32
    T_in = 34
    
    # We want to measure the energy of processing ONE tile.
    # Isolated Layer 2 Tile
    isolated_l2 = "/deeper/src/isolated_l2.yaml"
    generate_virtual_prob("/app/Examples/VGG_Simba/VGG02/VGG02_layer02.yaml", isolated_l2, T_out, T_out)
    
    # Virtual Fused Layer 2 Tile (P=34, Q=34)
    virtual_l2 = "/deeper/src/virtual_l2.yaml"
    generate_virtual_prob("/app/Examples/VGG_Simba/VGG02/VGG02_layer02.yaml", virtual_l2, T_in, T_in)
    
    print("Evaluating Baseline Tile (Isolated Layer 2)")
    e_iso, c_iso = run_timeloop(isolated_l2, "isolated")
    
    print("Evaluating Fused Tile (Virtual Layer 2)")
    e_fused, c_fused = run_timeloop(virtual_l2, "fused")
    
    if e_iso is None or e_fused is None:
        return
        
    # Extrapolate to full image (224x224 = 49 tiles of 32x32)
    # Note: Isolated needs DRAM read for Input, Virtual saves it.
    num_tiles = (224 // 32) * (224 // 32) # 49
    
    # Simplified macro-analysis
    total_e_iso = e_iso * num_tiles
    total_e_fused = e_fused * num_tiles
    
    print(f"\nExtrapolated Full Image Energy:")
    print(f"Isolated: {total_e_iso:.2f} uJ")
    print(f"Fused:    {total_e_fused:.2f} uJ")
    
    # Plotting
    labels = ['Isolated (Standard)', 'Fused (Analytical)']
    energies = [total_e_iso, total_e_fused]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, energies, color=['red', 'green'])
    plt.ylabel('Energy (uJ)')
    plt.title('Analytical Fusion vs. Isolated Execution')
    for i, v in enumerate(energies):
        plt.text(i, v + 1, f"{v:.1f}", ha='center', fontweight='bold')
    plt.savefig("/deeper/src/fusion_results.png")
    print("Saved graph to /deeper/src/fusion_results.png")

if __name__ == "__main__":
    main()
