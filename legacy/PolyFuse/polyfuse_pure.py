"""
PolyFuse Pure Analytical Optimizer
===================================
Zero-search algebraic tiling + dynamic programming for layer fusion
on rigid spatial DNN accelerators (Simba-like).

This script does NOT use any pre-computed DeepFrack logs, CheatSheet.json,
or BenchMarkLogFiles. It derives the optimal fused-layer partition and
tile sizes purely from the layer YAML definitions and hardware buffer sizes.

The analytical energy proxy uses a simple cost model:
  - MAC operation:  1 pJ per MAC
  - SRAM access:    5 pJ per element
  - DRAM access:  200 pJ per element
"""

import os
import yaml
import math
import time


def load_layers(layer_dir, layer_files=None):
    """Load CNN layer definitions from YAML files."""
    if layer_files is None:
        layer_files = sorted([f for f in os.listdir(layer_dir) if f.endswith('.yaml')
                              and 'New' not in f])
    layers = []
    for f in layer_files:
        with open(os.path.join(layer_dir, f)) as fh:
            d = yaml.safe_load(fh)
        inst = d['problem']['instance']
        layers.append({
            'C': inst.get('C', 1), 'M': inst.get('M', 1),
            'P': inst.get('P', 1), 'Q': inst.get('Q', 1),
            'R': inst.get('R', 1), 'S': inst.get('S', 1),
            'Wstride': inst.get('Wstride', 1),
            'Hstride': inst.get('Hstride', 1),
            'file': os.path.join(layer_dir, f),
            'name': f,
        })
    return layers


def compute_tile_sizes(layers, start, end, T_end):
    """
    Compute the output tile size at each layer and the input tile size
    to the start layer, given an output tile size T_end at the end layer.

    Returns:
        T_out: dict mapping layer index -> output tile size
        T_in_start: the input tile size that must be fed to the start layer
    """
    T_out = {}
    curr_T = T_end
    for i in range(end, start - 1, -1):
        T_out[i] = curr_T
        if i > start:
            # The input to layer i is the output of layer i-1
            # input_size = (output_size - 1) * stride + kernel_size
            curr_T = (curr_T - 1) * layers[i]['Wstride'] + layers[i]['R']

    # One more expansion to get the actual INPUT to the start layer
    T_in_start = (T_out[start] - 1) * layers[start]['Wstride'] + layers[start]['R']

    return T_out, T_in_start


def calc_analytical_cost(layers, start, end, T_end, cached_layers):
    """
    Estimate the energy cost of executing a fused stack of layers
    [start..end] with output tile size T_end, using a simple analytical
    energy model.

    The model accounts for:
    - MAC compute energy
    - Weight DRAM/SRAM traffic (with optional weight caching)
    - Input activation traffic (DRAM for start layer, SRAM for fused layers)
    - Output activation traffic (DRAM for end layer, SRAM for fused layers)
    - Halo recomputation (implicit via tile count * expanded tile size)
    """
    N_tiles = math.ceil(layers[end]['P'] / T_end) * math.ceil(layers[end]['Q'] / T_end)
    T_out, T_in_start = compute_tile_sizes(layers, start, end, T_end)

    energy_pJ = 0.0

    for i in range(start, end + 1):
        # --- Compute (MAC) energy ---
        # MACs per tile = T_out[i]^2 * R * S * C * M
        # Total MACs = MACs_per_tile * N_tiles
        macs_per_tile = (T_out[i] ** 2) * layers[i]['R'] * layers[i]['S'] * layers[i]['C'] * layers[i]['M']
        energy_pJ += macs_per_tile * N_tiles * 1.0

        # --- Weight traffic ---
        w_vol = layers[i]['R'] * layers[i]['S'] * layers[i]['C'] * layers[i]['M']
        if i in cached_layers:
            # Cached: fetch once from DRAM, read from SRAM for each tile
            energy_pJ += w_vol * 200.0              # one DRAM fetch
            energy_pJ += (w_vol * N_tiles) * 5.0    # N SRAM reads
        else:
            # Not cached: fetch from DRAM for each tile
            energy_pJ += (w_vol * N_tiles) * 200.0  # N DRAM fetches
            energy_pJ += (w_vol * N_tiles) * 5.0    # N SRAM reads

        # --- Input activation traffic ---
        if i == start:
            # Start layer reads its input from DRAM
            # Input size = T_in_start (the actual input, not the output)
            in_vol_per_tile = (T_in_start ** 2) * layers[i]['C']
            energy_pJ += in_vol_per_tile * N_tiles * 200.0
        else:
            # Fused intermediate: read from on-chip SRAM
            # Input to layer i = output of layer i-1
            in_T = (T_out[i] - 1) * layers[i]['Wstride'] + layers[i]['R']
            in_vol_per_tile = (in_T ** 2) * layers[i]['C']
            energy_pJ += in_vol_per_tile * N_tiles * 5.0

        # --- Output activation traffic ---
        out_vol_per_tile = (T_out[i] ** 2) * layers[i]['M']
        if i == end:
            # End layer writes output to DRAM
            energy_pJ += out_vol_per_tile * N_tiles * 200.0
        else:
            # Fused intermediate: write to on-chip SRAM
            energy_pJ += out_vol_per_tile * N_tiles * 5.0

    return energy_pJ / 1e6  # convert to uJ


def get_optimal_stack(layers, start, end, input_buffer, weight_buffer):
    """
    Find the optimal tile size and weight caching configuration for
    a fused stack [start..end].

    Returns:
        (best_cost, best_T_end, best_cache)
    """
    best_cost = float('inf')
    best_T_end = 1
    best_cache = []

    for T_end in range(1, layers[end]['P'] + 1):
        T_out, T_in_start = compute_tile_sizes(layers, start, end, T_end)

        # Check if input to start layer fits in the input buffer
        input_footprint = (T_in_start ** 2) * layers[start]['C']
        if input_footprint > input_buffer:
            break  # larger T_end will only make it worse

        # Try all weight caching combinations (2^n for n layers in stack)
        w_sizes = {i: layers[i]['R'] * layers[i]['S'] * layers[i]['C'] * layers[i]['M']
                   for i in range(start, end + 1)}
        n_layers = end - start + 1

        for comb in range(1 << n_layers):
            c_layers = []
            c_w = 0
            for bit in range(n_layers):
                if comb & (1 << bit):
                    l_idx = start + bit
                    c_layers.append(l_idx)
                    c_w += w_sizes[l_idx]
            if c_w <= weight_buffer:
                cost = calc_analytical_cost(layers, start, end, T_end, c_layers)
                if cost < best_cost:
                    best_cost = cost
                    best_T_end = T_end
                    best_cache = list(c_layers)

    return best_cost, best_T_end, best_cache


def optimize(layers, input_buffer=65536, weight_buffer=524288):
    """
    Run dynamic programming to find the optimal partition of layers
    into fused stacks.

    Returns:
        stacks: list of (start, end) tuples
        stack_info: dict of (start,end) -> (cost, T_end, cached_layers)
        total_cost: total analytical energy
        elapsed_ms: optimization time in milliseconds
    """
    st = time.time()
    n = len(layers)

    # dp[i] = minimum cost to process layers 0..i
    dp = [float('inf')] * n
    partition = [0] * n
    stack_info = {}

    for i in range(n):
        for j in range(i + 1):
            cost, T_end, c_layers = get_optimal_stack(layers, j, i, input_buffer, weight_buffer)
            stack_info[(j, i)] = (cost, T_end, c_layers)

            prev = dp[j - 1] if j > 0 else 0
            if prev + cost < dp[i]:
                dp[i] = prev + cost
                partition[i] = j

    et = time.time()

    # Reconstruct partition
    curr = n - 1
    stacks = []
    while curr >= 0:
        start = partition[curr]
        stacks.append((start, curr))
        curr = start - 1
    stacks.reverse()

    total_cost = sum(stack_info[s][0] for s in stacks)
    elapsed_ms = (et - st) * 1000

    return stacks, stack_info, total_cost, elapsed_ms


def main():
    layer_dir = '/app/Examples/AlexNet_Simba/AlexNet'
    layer_files = ["AlexNet_layer01.yaml", "AlexNet_layer02.yaml",
                   "AlexNet_layer03.yaml", "AlexNet_layer04.yaml"]

    layers = load_layers(layer_dir, layer_files)

    stacks, stack_info, total_cost, elapsed_ms = optimize(layers)

    print("=" * 60)
    print(" PURE POLYFUSE (NO EXTERNAL LOGS)")
    print("=" * 60)
    print(f"Total Analytical Optimization Time: {elapsed_ms:.2f} ms\n")

    for s in stacks:
        cost, T_end, c_layers = stack_info[s]
        T_out, T_in = compute_tile_sizes(layers, s[0], s[1], T_end)
        N_tiles = math.ceil(layers[s[1]]['P'] / T_end) * math.ceil(layers[s[1]]['Q'] / T_end)
        print(f"Fuse Stack: layers {s}")
        print(f"  Optimal T_end: {T_end}")
        print(f"  N_tiles: {N_tiles}")
        print(f"  Input tile to layer {s[0]}: {T_in}x{T_in}")
        for i in range(s[0], s[1] + 1):
            print(f"  Layer {i} output tile: {T_out[i]}x{T_out[i]}")
        print(f"  Cached Layers: {c_layers if c_layers else 'None'}")
        print(f"  Analytical Proxy Cost: {cost:.2f} uJ\n")

    print(f"Total Proxy Cost: {total_cost:.2f} uJ")

    # Also compute naive (no fusion) baseline for comparison
    naive_cost = 0
    print(f"\n{'='*60}")
    print(" NAIVE BASELINE (No Fusion)")
    print(f"{'='*60}")
    for i in range(len(layers)):
        cost_i, _, _ = get_optimal_stack(layers, i, i, 65536, 524288)
        naive_cost += cost_i
        print(f"  Layer {i}: {cost_i:.2f} uJ")
    print(f"  Total Naive Cost: {naive_cost:.2f} uJ")
    print(f"\n  Fusion Reduction: {(naive_cost - total_cost) / naive_cost * 100:.1f}%")


if __name__ == "__main__":
    main()
