import os
import yaml
import math
import time

def load_layers(layer_dir):
    files = ["AlexNet_layer01.yaml", "AlexNet_layer02.yaml", "AlexNet_layer03.yaml", "AlexNet_layer04.yaml"]
    layers = []
    for f in files:
        with open(os.path.join(layer_dir, f)) as file:
            d = yaml.safe_load(file)
        inst = d['problem']['instance']
        layers.append({
            'C': inst.get('C', 1), 'M': inst.get('M', 1),
            'P': inst.get('P', 1), 'Q': inst.get('Q', 1),
            'R': inst.get('R', 1), 'S': inst.get('S', 1),
            'Wstride': inst.get('Wstride', 1),
            'file': os.path.join(layer_dir, f)
        })
    return layers

def calc_analytical_cost(layers, start, end, T_end, cached_layers):
    N_tiles = math.ceil(layers[end]['P'] / T_end) * math.ceil(layers[end]['Q'] / T_end)
    
    T = {}
    curr_T = T_end
    for i in range(end, start - 1, -1):
        T[i] = curr_T
        if i > start:
            curr_T = (curr_T - 1) * layers[i]['Wstride'] + layers[i]['R']
    T[start-1] = curr_T 

    energy_pJ = 0
    for i in range(start, end + 1):
        macs = (T[i]**2) * layers[i]['R'] * layers[i]['S'] * layers[i]['C'] * layers[i]['M']
        energy_pJ += macs * 1.0
        
        w_vol = layers[i]['R'] * layers[i]['S'] * layers[i]['C'] * layers[i]['M']
        if i in cached_layers:
            energy_pJ += w_vol * 200.0  
            energy_pJ += (w_vol * N_tiles) * 5.0 
        else:
            energy_pJ += (w_vol * N_tiles) * 200.0 
            energy_pJ += (w_vol * N_tiles) * 5.0
            
        in_vol = (T[i-1]**2) * layers[i]['C'] * N_tiles
        if i == start:
            energy_pJ += in_vol * 200.0
        else:
            energy_pJ += in_vol * 5.0
            
        out_vol = (T[i]**2) * layers[i]['M'] * N_tiles
        if i == end:
            energy_pJ += out_vol * 200.0
        else:
            energy_pJ += out_vol * 5.0

    return energy_pJ / 1e6

def get_optimal_stack(layers, start, end):
    INPUT_BUFFER = 65536
    WEIGHT_BUFFER = 524288
    
    max_legal_T_end = 1
    best_cost = float('inf')
    best_cache = []
    
    for T_end in range(1, layers[end]['P'] + 1):
        curr_T = T_end
        for i in range(end, start - 1, -1):
            if i > start:
                curr_T = (curr_T - 1) * layers[i]['Wstride'] + layers[i]['R']
        T_in_start = curr_T
        
        if (T_in_start**2) * layers[start]['C'] > INPUT_BUFFER:
            break
        max_legal_T_end = T_end
        
        w_sizes = {i: layers[i]['R'] * layers[i]['S'] * layers[i]['C'] * layers[i]['M'] for i in range(start, end+1)}
        n_layers = end - start + 1
        for comb in range(1 << n_layers):
            c_layers = []
            c_w = 0
            for bit in range(n_layers):
                if (comb & (1 << bit)):
                    l_idx = start + bit
                    c_layers.append(l_idx)
                    c_w += w_sizes[l_idx]
            if c_w <= WEIGHT_BUFFER:
                cost = calc_analytical_cost(layers, start, end, T_end, c_layers)
                if cost < best_cost:
                    best_cost = cost
                    best_cache = c_layers

    return best_cost, max_legal_T_end, best_cache

def main():
    st = time.time()
    layers = load_layers('/app/Examples/AlexNet_Simba/AlexNet')
    n = len(layers)
    
    dp = [float('inf')] * n
    partition = [0] * n
    stack_info = {}
    
    for i in range(n):
        for j in range(i + 1):
            cost, T_end, c_layers = get_optimal_stack(layers, j, i)
            stack_info[(j, i)] = (cost, T_end, c_layers)
            
            prev = dp[j-1] if j > 0 else 0
            if prev + cost < dp[i]:
                dp[i] = prev + cost
                partition[i] = j
                
    et = time.time()
    print("=" * 60)
    print(" PURE POLYFUSE (NO EXTERNAL LOGS)")
    print("=" * 60)
    print(f"Total Analytical Optimization Time: {(et-st)*1000:.2f} ms\n")
    
    curr = n - 1
    stacks = []
    while curr >= 0:
        start = partition[curr]
        stacks.append((start, curr))
        curr = start - 1
    stacks.reverse()
    
    total_cost = 0
    for s in stacks:
        cost, T_end, c_layers = stack_info[s]
        total_cost += cost
        print(f"Fuse Stack: layers {s}")
        print(f"  Optimal T_end: {T_end}")
        print(f"  Cached Layers: {c_layers}")
        print(f"  Analytical Proxy Cost: {cost:.2f} uJ\n")
        
    print(f"Total Proxy Cost: {total_cost:.2f} uJ")
    
if __name__ == "__main__":
    main()
