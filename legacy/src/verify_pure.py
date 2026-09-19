import os
import yaml
import subprocess
import re
import math

def run_timeloop(problem_yaml, constraint_yaml, out_dir):
    arch_yaml = "/app/Examples/VGG_Simba/simba_like/arch/simba_like.yaml"
    comp_dir = "/app/Examples/VGG_Simba/simba_like/arch/components"
    components = [os.path.join(comp_dir, f) for f in os.listdir(comp_dir) if f.endswith('.yaml')]
    mapper_yaml = "/app/Examples/VGG_Simba/simba_like/mapper/mapper.yaml"
    
    cmd = ["/opt/timeloop/bin/timeloop-mapper", arch_yaml] + components + [mapper_yaml, constraint_yaml, problem_yaml]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/timeloop/lib:/opt/timeloop/build:" + env.get("LD_LIBRARY_PATH", "")
    
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(cmd, env=env, cwd=out_dir, capture_output=True, text=True)
    
    stats_file = os.path.join(out_dir, "timeloop-mapper.stats.txt")
    if not os.path.exists(stats_file):
        return None
    with open(stats_file, 'r') as f:
        content = f.read()
    energy_match = re.search(r'Energy\s*:\s*([0-9.]+)\s*uJ', content)
    return float(energy_match.group(1)) if energy_match else None

def generate_virtual_prob(base_prob, out_path, P, Q):
    with open(base_prob, 'r') as f:
        data = yaml.safe_load(f)
    data['problem']['instance']['P'] = P
    data['problem']['instance']['Q'] = Q
    with open(out_path, 'w') as f:
        yaml.dump(data, f)

def evaluate_stack(start, end, T_end, cached_layers):
    layer_dir = '/app/Examples/AlexNet_Simba/AlexNet'
    files = ["AlexNet_layer01.yaml", "AlexNet_layer02.yaml", "AlexNet_layer03.yaml", "AlexNet_layer04.yaml"]
    layers = []
    for f in files:
        with open(os.path.join(layer_dir, f)) as file:
            d = yaml.safe_load(file)
        layers.append({'data': d, 'file': os.path.join(layer_dir, f)})

    curr_T = T_end
    T_sizes = {}
    for i in range(end, start - 1, -1):
        T_sizes[i] = curr_T
        if i > start:
            inst = layers[i]['data']['problem']['instance']
            R = inst.get('R', 1)
            Wstride = inst.get('Wstride', 1)
            curr_T = (curr_T - 1) * Wstride + R
            
    print(f"  Validating Stack({start}, {end}) on Native Timeloop...")
    stack_cost = 0
    
    # Calculate N_tiles
    P_end = layers[end]['data']['problem']['instance']['P']
    Q_end = layers[end]['data']['problem']['instance']['Q']
    N_tiles = math.ceil(P_end / T_end) * math.ceil(Q_end / T_end)
    print(f"  N_tiles = {N_tiles}")
    
    for i in range(start, end + 1):
        if start == end:
            constr = 'SLC'
        else:
            if i == start:
                constr = 'OutWCC' if i in cached_layers else 'Start'
            elif i == end:
                constr = 'EWCC' if i in cached_layers else 'ELBLC'
            else:
                constr = 'WCC' if i in cached_layers else 'LBLC'
                
        prob_out = f"/deeper/src/tl_eval_polyfuse/stack_{start}_{end}_layer_{i}.yaml"
        out_dir = f"/deeper/src/tl_eval_polyfuse/out_{start}_{end}_{i}"
        os.makedirs(os.path.dirname(prob_out), exist_ok=True)
        generate_virtual_prob(layers[i]['file'], prob_out, T_sizes[i], T_sizes[i])
        
        constr_yaml = f"/app/Examples/VGG_Simba/simba_like/constraints/{constr}.yaml"
        e = run_timeloop(prob_out, constr_yaml, out_dir)
        
        if e is None:
            print(f"    Layer {i} FAILED natively! (Hardware constraint violated)")
            return None
        print(f"    Layer {i} ({constr}, T={T_sizes[i]}): {e} uJ")
        stack_cost += e
        
    return stack_cost * N_tiles

def main():
    import math
    print("======================================================")
    print(" NATIVE TIMELOOP HARDWARE VALIDATION")
    print("======================================================")
    # Stack (0, 2), T=13, Cache=[]
    e1 = evaluate_stack(0, 2, 13, [])
    if e1:
        print(f"  Stack (0, 2) Exact Native Energy: {e1} uJ")
        
    # Stack (3, 3), T=13, Cache=[]
    e2 = evaluate_stack(3, 3, 13, [])
    if e2:
        print(f"  Stack (3, 3) Exact Native Energy: {e2} uJ")
        
    if e1 and e2:
        print(f"\nTOTAL NATIVE POLYFUSE ENERGY: {e1 + e2} uJ")
        print("Hardware validation successful!")

if __name__ == "__main__":
    main()
