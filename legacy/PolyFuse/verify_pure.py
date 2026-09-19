"""
PolyFuse Timeloop Verification Script
======================================
Takes the output of the PolyFuse optimizer and validates each
fused stack by running the native Timeloop-mapper C++ simulator.

This script:
1. Generates virtual problem YAMLs with the tile sizes computed by PolyFuse
2. Runs timeloop-mapper with the appropriate fusion constraint for each layer
3. Multiplies the per-tile energy by N_tiles to get total energy
4. Also runs the Naive (SLC) baseline for comparison
"""

import os
import sys
import yaml
import subprocess
import re
import math
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polyfuse_pure import load_layers, optimize, compute_tile_sizes

# Paths
ARCH_YAML = "/app/Examples/VGG_Simba/simba_like/arch/simba_like.yaml"
COMP_DIR = "/app/Examples/VGG_Simba/simba_like/arch/components"
MAPPER_YAML = "/app/Examples/VGG_Simba/simba_like/mapper/mapper_fast.yaml"
CONSTRAINTS_DIR = "/app/Examples/VGG_Simba/simba_like/constraints"
TIMELOOP_BIN = "/opt/timeloop/bin/timeloop-mapper"
EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tl_eval")


def run_timeloop(problem_yaml, constraint_yaml, out_dir):
    """Run the Timeloop mapper and return the energy in uJ, or None on failure."""
    components = [os.path.join(COMP_DIR, f) for f in sorted(os.listdir(COMP_DIR))
                  if f.endswith('.yaml')]
    cmd = [TIMELOOP_BIN, ARCH_YAML] + components + [MAPPER_YAML, constraint_yaml, problem_yaml]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/timeloop/lib:/opt/timeloop/build:" + env.get("LD_LIBRARY_PATH", "")

    os.makedirs(out_dir, exist_ok=True)
    result = subprocess.run(cmd, env=env, cwd=out_dir, capture_output=True, text=True, timeout=1800)

    stats_file = os.path.join(out_dir, "timeloop-mapper.stats.txt")
    if not os.path.exists(stats_file):
        print(f"    WARNING: No stats file generated in {out_dir}")
        if result.stderr:
            print(f"    stderr: {result.stderr[:200]}")
        return None
    with open(stats_file, 'r') as f:
        content = f.read()
    energy_match = re.search(r'^Energy:\s*([0-9.]+)\s*uJ', content, re.MULTILINE)
    return float(energy_match.group(1)) if energy_match else None


def generate_virtual_prob(base_prob, out_path, P, Q):
    """Create a virtual problem YAML with modified P and Q (tile sizes)."""
    with open(base_prob, 'r') as f:
        data = yaml.safe_load(f)
    data['problem']['instance']['P'] = P
    data['problem']['instance']['Q'] = Q
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(data, f)


def get_constraint(start, end, i, cached_layers):
    """Determine the correct Timeloop constraint file for a layer in a stack."""
    if start == end:
        return 'SLC'
    if i == start:
        return 'OutWCC' if i in cached_layers else 'Start'
    elif i == end:
        return 'EWCC' if i in cached_layers else 'ELBLC'
    else:
        return 'WCC' if i in cached_layers else 'LBLC'


def evaluate_stack_native(layers, start, end, T_end, cached_layers, label=""):
    """
    Evaluate a fused stack using native Timeloop.
    Returns (total_energy, per_layer_energies) or (None, {}) on failure.
    """
    T_out, T_in_start = compute_tile_sizes(layers, start, end, T_end)
    N_tiles = math.ceil(layers[end]['P'] / T_end) * math.ceil(layers[end]['Q'] / T_end)

    print(f"\n  Evaluating {label}Stack({start},{end}), T_end={T_end}, N_tiles={N_tiles}")
    print(f"    Input tile to layer {start}: {T_in_start}x{T_in_start}")
    for i in range(start, end + 1):
        print(f"    Layer {i} output tile: {T_out[i]}x{T_out[i]}")

    per_layer = {}
    for i in range(start, end + 1):
        constr = get_constraint(start, end, i, cached_layers)
        constr_yaml = os.path.join(CONSTRAINTS_DIR, f"{constr}.yaml")

        prob_out = os.path.join(EVAL_DIR, f"stack_{start}_{end}_layer_{i}.yaml")
        out_dir = os.path.join(EVAL_DIR, f"out_{start}_{end}_{i}")

        generate_virtual_prob(layers[i]['file'], prob_out, T_out[i], T_out[i])
        e = run_timeloop(prob_out, constr_yaml, out_dir)

        if e is None:
            print(f"    Layer {i} ({constr}, T={T_out[i]}): FAILED")
            return None, {}
        per_layer[i] = e
        print(f"    Layer {i} ({constr}, T={T_out[i]}): {e:.2f} uJ/tile")

    per_tile_total = sum(per_layer.values())
    total = per_tile_total * N_tiles
    print(f"    Per-tile total: {per_tile_total:.2f} uJ × {N_tiles} tiles = {total:.2f} uJ")
    return total, per_layer


def main():
    layer_dir = '/app/Examples/AlexNet_Simba/AlexNet'
    layer_files = ["AlexNet_layer01.yaml", "AlexNet_layer02.yaml",
                   "AlexNet_layer03.yaml", "AlexNet_layer04.yaml"]

    layers = load_layers(layer_dir, layer_files)

    # Run the optimizer
    stacks, stack_info, total_analytical, elapsed_ms = optimize(layers)

    print("=" * 60)
    print(" POLYFUSE NATIVE TIMELOOP VERIFICATION")
    print("=" * 60)
    print(f"Optimizer found partition in {elapsed_ms:.2f} ms")
    print(f"Analytical proxy cost: {total_analytical:.2f} uJ")

    # Evaluate each fused stack natively
    print(f"\n--- POLYFUSE FUSED PARTITION ---")
    polyfuse_total = 0
    for s in stacks:
        cost_a, T_end, cached = stack_info[s]
        e, _ = evaluate_stack_native(layers, s[0], s[1], T_end, cached, label="PolyFuse ")
        if e is None:
            print(f"  Stack {s} FAILED native validation!")
            return
        polyfuse_total += e

    print(f"\n  POLYFUSE TOTAL: {polyfuse_total:.2f} uJ")

    # Evaluate naive baseline (each layer individually with SLC)
    print(f"\n--- NAIVE BASELINE (No Fusion) ---")
    naive_total = 0
    for i in range(len(layers)):
        T_end_naive = layers[i]['P']  # Full output size
        e, _ = evaluate_stack_native(layers, i, i, T_end_naive, [], label="Naive ")
        if e is None:
            print(f"  Layer {i} FAILED!")
            return
        naive_total += e

    print(f"\n  NAIVE TOTAL: {naive_total:.2f} uJ")

    # DeepFrack comparison (from their log file)
    # DeepFrack chose: Stack(0,1) + Stack(2,2) + Stack(3,3) = 2722.65 uJ (from JSON)
    print(f"\n--- DEEPFRACK (from their log, JSON-scale) ---")
    print(f"  DeepFrack total (JSON): 2722.65 uJ")
    print(f"  Naive total (JSON): 2908.83 uJ")
    print(f"  DeepFrack reduction: {(2908.83 - 2722.65) / 2908.83 * 100:.1f}%")

    # Summary
    print(f"\n{'='*60}")
    print(f" SUMMARY (Native Timeloop Scale)")
    print(f"{'='*60}")
    print(f"  Naive:    {naive_total:.2f} uJ")
    print(f"  PolyFuse: {polyfuse_total:.2f} uJ")
    reduction = (naive_total - polyfuse_total) / naive_total * 100
    print(f"  Reduction: {reduction:.1f}%")
    print(f"  Optimizer time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
