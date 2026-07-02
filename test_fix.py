import sys
sys.path.insert(0, '/deeper/PolyFuse')
from polyfuse_pure import load_layers, compute_tile_sizes, calc_analytical_cost

layers = load_layers('/app/Examples/AlexNet_Simba/AlexNet', ["AlexNet_layer01.yaml", "AlexNet_layer02.yaml", "AlexNet_layer03.yaml", "AlexNet_layer04.yaml"])

def get_optimal_stack_fixed(layers, start, end, input_buffer, weight_buffer):
    best_cost = float('inf')
    best_T_end = 1
    best_cache = []

    for T_end in range(1, layers[end]['P'] + 1):
        T_out, T_in_start = compute_tile_sizes(layers, start, end, T_end)

        valid = True
        # Check start layer input
        if (T_in_start ** 2) * layers[start]['C'] > input_buffer:
            valid = False
            
        for i in range(start, end + 1):
            # Check layer output buffer
            if (T_out[i] ** 2) * layers[i]['M'] > 65536: # using output_buffer which is also 65536
                valid = False
            # Check intermediate input buffer
            if i > start:
                t_in = (T_out[i]-1)*layers[i]['Wstride'] + layers[i]['R']
                if (t_in ** 2) * layers[i]['C'] > input_buffer:
                    valid = False
                    
        if not valid:
            continue # Try next T_end instead of breaking? Wait, larger T_end always means larger footprints, so we CAN break.
            
        w_sizes = {i: layers[i]['R'] * layers[i]['S'] * layers[i]['C'] * layers[i]['M'] for i in range(start, end + 1)}
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

def optimize_fixed(layers):
    n = len(layers)
    dp = [float('inf')] * n
    partition = [0] * n
    stack_info = {}

    for i in range(n):
        for j in range(i + 1):
            cost, T_end, c_layers = get_optimal_stack_fixed(layers, j, i, 65536, 524288)
            stack_info[(j, i)] = (cost, T_end, c_layers)

            prev = dp[j - 1] if j > 0 else 0
            if prev + cost < dp[i]:
                dp[i] = prev + cost
                partition[i] = j

    curr = n - 1
    stacks = []
    while curr >= 0:
        start = partition[curr]
        stacks.append((start, curr))
        curr = start - 1
    stacks.reverse()
    return stacks, stack_info

stacks, stack_info = optimize_fixed(layers)
print("FIXED PARTITION:")
for s in stacks:
    print(f"Stack {s}: T_end={stack_info[s][1]}, Cost={stack_info[s][0]:.2f} uJ")
