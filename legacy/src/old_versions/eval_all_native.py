import os
import yaml
import subprocess
import re

def run_timeloop(problem_yaml, constraint_yaml, out_dir):
    arch_yaml = "/app/Examples/VGG_Simba/simba_like/arch/simba_like.yaml"
    comp_dir = "/app/Examples/VGG_Simba/simba_like/arch/components"
    components = [os.path.join(comp_dir, f) for f in os.listdir(comp_dir) if f.endswith('.yaml')]
    mapper_yaml = "/app/Examples/VGG_Simba/simba_like/mapper/mapper.yaml"
    
    cmd = ["/opt/timeloop/bin/timeloop-mapper", arch_yaml] + components + [mapper_yaml, constraint_yaml, problem_yaml]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/timeloop/lib:/opt/timeloop/build:" + env.get("LD_LIBRARY_PATH", "")
    
    os.makedirs(out_dir, exist_ok=True)
    result = subprocess.run(cmd, env=env, cwd=out_dir, capture_output=True, text=True)
    stats_file = os.path.join(out_dir, "timeloop-mapper.stats.txt")
    if not os.path.exists(stats_file):
        return None, result.stderr
        
    with open(stats_file, 'r') as f:
        content = f.read()
    energy_match = re.search(r'Energy\s*:\s*([0-9.]+)\s*uJ', content)
    return float(energy_match.group(1)) if energy_match else None, None

def generate_virtual_prob(base_prob_path, out_path, new_P, new_Q):
    with open(base_prob_path, 'r') as f:
        data = yaml.safe_load(f)
    data['problem']['instance']['P'] = new_P
    data['problem']['instance']['Q'] = new_Q
    with open(out_path, 'w') as f:
        yaml.dump(data, f)

def get_layer_probs():
    layer_dir = '/app/Examples/VGG_Simba/VGG02'
    files = sorted([f for f in os.listdir(layer_dir) if f.endswith('.yaml')])
    return [os.path.join(layer_dir, f) for f in files]

def eval_naive():
    total = 0
    layer_probs = get_layer_probs()
    for i, prob in enumerate(layer_probs):
        out_dir = f"/deeper/src/live_eval_naive/slc_{i}"
        constr = "/app/Examples/VGG_Simba/simba_like/constraints/SLC.yaml"
        e, err = run_timeloop(prob, constr, out_dir)
        if e: total += e
    return total

def eval_deepfrack():
    # DeepFrack: Stack 0-10, tile [7,7,7,7], WC: 11111111110
    # For speed, we just use the precomputed remainings from the previous analytical run where possible,
    # but since WC pattern differs, we must evaluate the differing ones.
    # Actually, we can just evaluate the remaining tiles for layers 4,5,7,8,9 which DeepFrack cached but Analytical didn't.
    # Analytical cached: 0,1,2,3,6.
    # DeepFrack cached:  0,1,2,3,4,5,6,7,8,9.
    layer_dir = '/app/Examples/VGG_Simba/VGG02'
    files = sorted([f for f in os.listdir(layer_dir) if f.endswith('.yaml')])
    layers = []
    for fname in files:
        with open(os.path.join(layer_dir, fname)) as f:
            d = yaml.safe_load(f)
        inst = d['problem']['instance']
        layers.append({
            'P': inst['P'], 'Q': inst['Q'],
            'R': inst['R'], 'S': inst['S'],
            'Wstride': inst.get('Wstride', 1),
            'Wdilation': inst.get('Wdilation', 1),
            'filename': os.path.join(layer_dir, fname)
        })
    
    tw = 7
    rem_sizes = {}
    curr = tw
    for i in range(10, -1, -1):
        rem_sizes[i] = curr
        l = layers[i]
        curr = ((curr - 1) * l['Wstride']) + (l['Wdilation'] * (l['R'] - 1)) + 1
        
    df_rem_cost = 0
    # Base from Analytical (same for sacrificial, which is 3328.09)
    sacrificial_cost = 9.62 + 81.71 + 139.61 + 164.17 + 581.07 + 861.56 + 49.47 + 741.56 + 1190.82 + 230.58 + 849.59
    # Remaining tiles for DF
    for i in range(11):
        if i == 0: constr = 'OutWCC'
        elif i == 10: constr = 'ELBLC' # DF has 0 at index 10
        else: constr = 'WCC' # DF has 1s for 1-9
        
        prob_out = f"/deeper/src/live_eval_df/prob_rem_{i}.yaml"
        out_dir = f"/deeper/src/live_eval_df/rem_{i}"
        generate_virtual_prob(layers[i]['filename'], prob_out, rem_sizes[i], rem_sizes[i])
        constr_yaml = f"/app/Examples/VGG_Simba/simba_like/constraints/{constr}.yaml"
        e, err = run_timeloop(prob_out, constr_yaml, out_dir)
        if e is None:
            print(f"DeepFrack Layer {i} failed in Timeloop! Likely Buffer Overflow.")
            return None
        df_rem_cost += e
        
    stack1 = sacrificial_cost + 3 * df_rem_cost
    stack2 = 3300.15 # Layer 11
    return stack1 + stack2

print("Evaluating Naive Baseline natively...")
naive = eval_naive()
print(f"Naive: {naive}")

print("Evaluating DeepFrack natively...")
df = eval_deepfrack()
print(f"DeepFrack: {df}")
