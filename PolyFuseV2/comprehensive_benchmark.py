import os
import subprocess
import glob
import re

networks = [
    {
        "name": "AlexNet",
        "path": "/app/Examples/AlexNet_Simba/AlexNet/",
        "arch_dir": "/app/Examples/AlexNet_Simba/simba_like"
    },
    {
        "name": "VGG02",
        "path": "/app/Examples/VGG_Simba/VGG02/",
        "arch_dir": "/app/Examples/AlexNet_Simba/simba_like"
    },
    {
        "name": "MobileNet",
        "path": "/app/Examples/MobileNet_SimbaSystolic/MobileNet/",
        "arch_dir": "/app/Examples/AlexNet_Simba/simba_like"
    }
]

COMPONENTS_DIR = "/deeper/PolyFuseV2/data/components"
TIMELOOP_MAPPER = "/tmp/accelergy-timeloop-infrastructure/src/timeloop/build/timeloop-mapper"
OUTPUT_DIR = "/deeper/PolyFuseV2/comprehensive_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def run_timeloop_naive(network):
    import shutil
    print(f"\n[Timeloop Baseline] Running for {network['name']}...")
    yaml_files = sorted(glob.glob(os.path.join(network['path'], "*.yaml")), key=natural_sort_key)
    
    # New directory in which everything is saved
    benchmark_runs_dir = "/deeper/PolyFuseV2/benchmark_runs"
    naive_runs_dir = os.path.join(benchmark_runs_dir, "naive")
    os.makedirs(naive_runs_dir, exist_ok=True)
    
    total_energy_pj = 0.0
    for layer_yaml in yaml_files:
        layer_name = os.path.basename(layer_yaml).replace('.yaml', '')
        print(f"  -> Evaluating {layer_name}...")
        
        # Clean old stats
        stats_file = '/tmp/timeloop-mapper.stats.txt'
        if os.path.exists(stats_file):
            os.remove(stats_file)
            
        cmd = [
            TIMELOOP_MAPPER,
            os.path.join(network['arch_dir'], "arch/simba_like.yaml"),
            layer_yaml,
            os.path.join(network['arch_dir'], "mapper/mapper.yaml"),
            os.path.join(COMPONENTS_DIR, "simba_like_map_constraints.yaml"),
            os.path.join(COMPONENTS_DIR, "simba_like_arch_constraints.yaml"),
            os.path.join(COMPONENTS_DIR, "lmac.yaml"),
            os.path.join(network['arch_dir'], "arch/components/reg_storage.yaml"),
            os.path.join(network['arch_dir'], "arch/components/smartbuffer_RF.yaml"),
            os.path.join(network['arch_dir'], "arch/components/smartbuffer_SRAM.yaml")
        ]
        
        subprocess.run(cmd, cwd='/tmp', capture_output=True, text=True)
        
        if os.path.exists(stats_file):
            # Save the stats file under the new directory
            saved_stats_path = os.path.join(naive_runs_dir, f"{network['name']}_{layer_name}_stats.txt")
            shutil.copy(stats_file, saved_stats_path)
            
            with open(stats_file, 'r') as f:
                for line in f:
                    if "Energy:" in line:
                        val = float(line.split()[1])
                        line_lower = line.lower()
                        if "pj" in line_lower:
                            total_energy_pj += val
                        elif "nj" in line_lower:
                            total_energy_pj += (val * 1e3)
                        elif "uj" in line_lower:
                            total_energy_pj += (val * 1e6)
                        elif "mj" in line_lower:
                            total_energy_pj += (val * 1e9)
                        elif " j" in line_lower or line_lower.strip().endswith(" j"):
                            total_energy_pj += (val * 1e12)
                        break
        else:
            print(f"     [ERROR] Stats file missing for {layer_name}")
            
    print(f"  [Timeloop Baseline] {network['name']} Total: {total_energy_pj / 1e9:.2f} mJ")
    return total_energy_pj

def run_polyfuse_pipeline(network):
    import shutil
    print(f"\n[PolyFuse Pipeline] Running for {network['name']}...")
    cmd = [
        "python3", "-m", "polyfuse.cli",
        "--arch", os.path.join(network['arch_dir'], "arch/simba_like.yaml"),
        "--components", COMPONENTS_DIR,
        "--network", network['path'],
        "--output", OUTPUT_DIR,
        "--verbose"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Save the full output directory to the benchmark_runs directory!
    benchmark_runs_dir = "/deeper/PolyFuseV2/benchmark_runs"
    polyfuse_dest_dir = os.path.join(benchmark_runs_dir, f"polyfuse_{network['name']}")
    if os.path.exists(polyfuse_dest_dir):
        shutil.rmtree(polyfuse_dest_dir)
    if os.path.exists(OUTPUT_DIR):
        shutil.copytree(OUTPUT_DIR, polyfuse_dest_dir)
    
    polyfuse_pj = 0.0
    deepfrack_pj = 0.0
    
    for line in result.stdout.splitlines():
        if "PolyFuse (optimal DP)" in line:
            parts = line.split()
            polyfuse_pj = float(parts[3])
        if "DeepFrack (fuse all)" in line:
            parts = line.split()
            deepfrack_pj = float(parts[3])
            
    if polyfuse_pj == 0.0:
        print("  [ERROR] PolyFuse did not output a valid energy!")
        print("STDOUT:", result.stdout)
        
    print(f"  [PolyFuse] {network['name']} Total: {polyfuse_pj / 1e9:.2f} mJ")
    print(f"  [DeepFrack] {network['name']} Total: {deepfrack_pj / 1e9:.2f} mJ")
    return polyfuse_pj, deepfrack_pj

results = []

for net in networks:
    print(f"============================================================")
    print(f" EVALUATING NETWORK: {net['name']}")
    print(f"============================================================")
    
    timeloop_energy_pj = run_timeloop_naive(net)
    polyfuse_pj, deepfrack_pj = run_polyfuse_pipeline(net)
    
    results.append({
        "name": net["name"],
        "naive_mj": timeloop_energy_pj / 1e9,
        "polyfuse_mj": polyfuse_pj / 1e9,
        "deepfrack_mj": deepfrack_pj / 1e9
    })

print("\n\n")
print("============================================================")
print("  COMPREHENSIVE BENCHMARK RESULTS (SIMBA ARCHITECTURE)")
print("============================================================")
print(f"{'Network':<15} | {'Naive (mJ)':<12} | {'PolyFuse (mJ)':<13} | {'PolyFuse Speedup':<16} | {'DeepFrack (mJ)':<14}")
print("-" * 80)
for r in results:
    poly_speedup = (r['naive_mj'] / r['polyfuse_mj']) if r['polyfuse_mj'] > 0 else 0.0
    print(f"{r['name']:<15} | {r['naive_mj']:<12.2f} | {r['polyfuse_mj']:<13.2f} | {poly_speedup:<16.2f} | {r['deepfrack_mj']:<14.2f}")
print("============================================================")
