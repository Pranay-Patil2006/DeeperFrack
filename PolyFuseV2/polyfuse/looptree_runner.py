"""
LoopTree runner for PolyFuse V2.

Generates the symbolic LoopTree mapping for fused or unfused layer blocks
and evaluates them using the pytimeloop LoopTree engine in a subprocess.

KEY FIX: All evaluations now use the REAL Simba architecture (passed in via
hw_config.arch_path), NOT a generic 3-level abstraction. The mapping includes
both temporal tiling AND spatial unrolling across the PE array.
"""
import os
import sys
import yaml
from pathlib import Path

sys.path.insert(0, "/tmp/accelergy-timeloop-infrastructure/src/timeloop-python")

from pytimeloop.looptree.run import run_looptree


# -----------------------------------------------------------------------
# Fused Workload YAML generation
# -----------------------------------------------------------------------

def generate_fused_workload(layers, layer_indices):
    """
    Generate a Timeloop v4-fused workload YAML for a stack of layers.
    Each layer's output tensor is the next layer's input tensor (producer/consumer).

    Projections must be lists-of-lists, e.g.:
        [[C], [M], [R], [S]]     for Weights
        [[N], [C], [Q, S], [P, R]]  for Inputs (stride folded into a sum)
    """
    problem_list = []

    for i, (layer, orig_idx) in enumerate(zip(layers, layer_indices)):
        c       = layer.get('C', 1)
        m       = layer.get('M', 1)
        p       = layer.get('P', 1)
        q       = layer.get('Q', 1)
        r       = layer.get('R', 1)
        s       = layer.get('S', 1)
        wstride = layer.get('Wstride', 1)
        hstride = layer.get('Hstride', 1)

        shape_name  = f"Conv_{orig_idx}"
        fmap_in     = f"Inter_{i-1}" if i > 0 else "Inputs"
        fmap_out    = f"Inter_{i}" if i < len(layers)-1 else "Outputs"
        filter_name = f"Weights_{orig_idx}"

        c_dim = f"C_{orig_idx}"
        m_dim = f"M_{orig_idx}"
        p_dim = f"P_{orig_idx}"
        q_dim = f"Q_{orig_idx}"
        r_dim = f"R_{orig_idx}"
        s_dim = f"S_{orig_idx}"
        n_dim = f"N_{orig_idx}"

        # The pytimeloop LoopTree parser expects projections as a string in
        # the format '[ dim1, dim2 ]' where each element is a loop dimension.
        # For affine sums (stride > 1), the format uses arithmetic expressions:
        #   '[ N, C, hstride*Q+S, wstride*P+R ]'
        # This matches the official cascaded_mm.workload.yaml in pytimeloop tests.
        
        weights_proj = f"[ {c_dim}, {m_dim}, {r_dim}, {s_dim} ]"
        
        if hstride == 1 and wstride == 1:
            inputs_proj = f"[ {n_dim}, {c_dim}, {q_dim}+{s_dim}, {p_dim}+{r_dim} ]"
        else:
            inputs_proj = f"[ {n_dim}, {c_dim}, {hstride}*{q_dim}+{s_dim}, {wstride}*{p_dim}+{r_dim} ]"
        
        outputs_proj = f"[ {n_dim}, {m_dim}, {q_dim}, {p_dim} ]"

        shape_def = {
            'name': shape_name,
            'dimensions': [c_dim, m_dim, r_dim, s_dim, n_dim, p_dim, q_dim],
            'data_spaces': [
                {
                    'name': filter_name,
                    'dimensions': [f'{filter_name}_C', f'{filter_name}_M',
                                   f'{filter_name}_R', f'{filter_name}_S'],
                    'projection': weights_proj,
                },
                {
                    'name': fmap_in,
                    'dimensions': [f'{fmap_in}_N', f'{fmap_in}_C',
                                   f'{fmap_in}_H', f'{fmap_in}_W'],
                    'projection': inputs_proj,
                },
                {
                    'name': fmap_out,
                    'dimensions': [f'{fmap_out}_N', f'{fmap_out}_M',
                                   f'{fmap_out}_H', f'{fmap_out}_W'],
                    'projection': outputs_proj,
                    'read_write': True,
                },
            ],
        }

        instance_str = (
            f"0 <= {c_dim} < {c} and 0 <= {m_dim} < {m} and "
            f"0 <= {r_dim} < {r} and 0 <= {s_dim} < {s} and "
            f"0 <= {n_dim} < 1 and 0 <= {p_dim} < {p} and 0 <= {q_dim} < {q}"
        )

        problem_list.append({'shape': shape_def, 'instance': instance_str})

    return yaml.dump({'problem': problem_list}, default_flow_style=False)



# -----------------------------------------------------------------------
# Symbolic Mapping YAML generation
# -----------------------------------------------------------------------

def get_target_mapping(arch_path):
    if not arch_path or not os.path.exists(arch_path):
        return {'DRAM': 0, 'GlobalBuffer': 1, 'PEInputBuffer': 2, 'PEWeightBuffer': 3, 'PEAccuBuffer': 4, 'PEWeightRegs': 5, 'LMAC': 6}
    import yaml
    components = []
    
    def dfs(node):
        if isinstance(node, dict):
            if 'local' in node:
                for comp in node['local']:
                    name = comp['name'].split('[')[0]
                    components.append(name)
            if 'subtree' in node:
                for child in node['subtree']:
                    dfs(child)
        elif isinstance(node, list):
            for item in node:
                dfs(item)
                
    with open(arch_path, 'r') as f:
        arch_dict = yaml.safe_load(f)
        if 'architecture' in arch_dict:
            dfs(arch_dict['architecture'].get('subtree', []))
        else:
            dfs(arch_dict.get('subtree', []))
            
    return {name: i for i, name in enumerate(components)}


def generate_symbolic_mapping(layers, layer_indices, tile_sizes, hw_config=None):
    """
    Generate the LoopTree mapping structure for a given set of layers dynamically.
    Resolves targets using the actual parsed hardware config roles to support arbitrary graphs.
    """
    nodes = []

    all_filters   = [f"Weights_{i}" for i in layer_indices]
    intermediates = [f"Inter_{i}" for i in range(len(layers) - 1)]
    fmap_in  = "Inputs"
    fmap_out = "Outputs"

    arch_path = hw_config.arch_path if hw_config else None
    t_map = get_target_mapping(arch_path)
    
    # Dynamic role-to-name resolver from parsed hardware configuration
    dram_name = None
    gb_name = None
    in_buf_name = None
    wt_buf_name = None
    acc_name = None
    wt_reg_name = None
    
    if hw_config and hw_config.buffers:
        for name, info in hw_config.buffers.items():
            role = info.get('role')
            if role == 'dram':
                dram_name = name
            elif role == 'global':
                gb_name = name
            elif role == 'input':
                in_buf_name = name
            elif role == 'weight':
                wt_buf_name = name
            elif role == 'weight_reg':
                wt_reg_name = name
            elif role == 'output':
                acc_name = name

    # Resolve fallbacks for shared/custom hierarchies
    if dram_name is None:
        dram_name = 'DRAM'
    if gb_name is None:
        gb_name = 'GlobalBuffer'
    if in_buf_name is None:
        in_buf_name = gb_name
    if wt_buf_name is None:
        wt_buf_name = gb_name
    if acc_name is None:
        acc_name = gb_name

    # Find compute units dynamically in target map
    mac_name = 'LMAC'
    for name in t_map:
        name_lower = name.lower()
        if 'mac' in name_lower or 'compute' in name_lower:
            mac_name = name
            break

    target_dram = t_map.get(dram_name, 0)
    target_gb = t_map.get(gb_name, 1)
    target_in_buf = t_map.get(in_buf_name, 2)
    target_wt_buf = t_map.get(wt_buf_name, 3)
    target_acc = t_map.get(acc_name, 4)
    target_wt_reg = t_map.get(wt_reg_name, 5) if wt_reg_name else None
    target_mac = t_map.get(mac_name, 6)

    dram_dspace = all_filters + [fmap_in, fmap_out]
    nodes.append({'type': 'storage', 'target': target_dram, 'dspace': dram_dspace})

    gb_dspace = all_filters + intermediates + [fmap_in, fmap_out]
    nodes.append({'type': 'storage', 'target': target_gb, 'dspace': gb_dspace})

    last_idx  = layer_indices[-1]
    last_name = layers[-1].get('name', f'layer{last_idx}')

    pe_count = hw_config.pe_array.get('total_pes', 1) if hw_config else 1
    t_out_last = tile_sizes.get(last_name, {}).get('T_out', 1) if tile_sizes else 1

    nodes.append({'type': 'temporal', 'rank': f'N_{last_idx}', 'tile_shape': 1})
    nodes.append({'type': 'temporal', 'rank': f'P_{last_idx}', 'tile_shape': t_out_last})
    nodes.append({'type': 'temporal', 'rank': f'Q_{last_idx}', 'tile_shape': t_out_last})

    branches = []
    for i, (layer, orig_idx) in enumerate(zip(layers, layer_indices)):
        name = layer.get('name', f'layer{orig_idx}')
        ts   = tile_sizes.get(name, {}) if tile_sizes else {}
        t_c  = ts.get('T_c', 1)
        t_m  = ts.get('T_m', 1)
        C    = layer.get('C', 1)
        M    = layer.get('M', 1)
        R    = layer.get('R', 1)
        S    = layer.get('S', 1)

        import math
        
        # Simba specifically unrolls C and M spatially (typically 4x4 array)
        spatial_c = int(math.isqrt(pe_count)) if pe_count > 1 else 1
        spatial_m = pe_count // spatial_c if pe_count > 1 else 1
        
        inner_c = math.ceil(t_c / spatial_c)
        inner_m = math.ceil(t_m / spatial_m)
        
        c_outer = math.ceil(C / (inner_c * spatial_c))
        m_outer = math.ceil(M / (inner_m * spatial_m))

        is_first = (i == 0)
        is_last  = (i == len(layers) - 1)
        
        in_tensor  = "Inputs" if is_first else f"Inter_{i - 1}"
        out_tensor = "Outputs" if is_last else f"Inter_{i}"

        branch = [
            {'type': 'temporal', 'rank': f'C_{orig_idx}', 'tile_shape': c_outer},
            {'type': 'temporal', 'rank': f'M_{orig_idx}', 'tile_shape': m_outer},
            
            # SPATIAL UNROLLING (outside local PE buffers)
            {'type': 'spatial', 'rank': f'C_{orig_idx}', 'tile_shape': spatial_c},
            {'type': 'spatial', 'rank': f'M_{orig_idx}', 'tile_shape': spatial_m},
            
            {'type': 'storage', 'target': target_acc, 'dspace': [out_tensor]},
            {'type': 'storage', 'target': target_wt_buf, 'dspace': [f"Weights_{orig_idx}"]},
            {'type': 'storage', 'target': target_in_buf, 'dspace': [in_tensor]},
            
            # TEMPORAL (inner per-PE)
            {'type': 'temporal', 'rank': f'C_{orig_idx}', 'tile_shape': inner_c},
            {'type': 'temporal', 'rank': f'M_{orig_idx}', 'tile_shape': inner_m},
            {'type': 'temporal', 'rank': f'R_{orig_idx}', 'tile_shape': R},
            {'type': 'temporal', 'rank': f'S_{orig_idx}', 'tile_shape': S},
        ]
        
        if wt_reg_name and wt_reg_name in t_map:
            branch.append({'type': 'storage', 'target': target_wt_reg, 'dspace': [f"Weights_{orig_idx}"]})
            
        branch.append({'type': 'compute', 'einsum': f"Conv_{orig_idx}", 'target': target_mac})
        branches.append(branch)

    nodes.append({'type': 'sequential', 'branches': branches})

    return yaml.dump({'mapping': {'type': 'fused', 'nodes': nodes}})


# -----------------------------------------------------------------------
# Subprocess evaluation
# -----------------------------------------------------------------------

def evaluate_single_block(layers, layer_indices, tile_sizes, tmp_dir, hw_config=None):
    """
    Evaluate a single fused block using LoopTree in an isolated subprocess.

    The subprocess uses the REAL Simba arch file (from hw_config.arch_path)
    rather than the generic 3-level abstraction. This ensures energy values
    reflect the actual memory hierarchy energy costs from Accelergy.
    """
    import subprocess

    os.makedirs(tmp_dir, exist_ok=True)

    workload_yaml = generate_fused_workload(layers, layer_indices)
    mapping_yaml  = generate_symbolic_mapping(layers, layer_indices,
                                              tile_sizes, hw_config)

    with open(os.path.join(tmp_dir, 'workload.yaml'), 'w') as f:
        f.write(workload_yaml)
    with open(os.path.join(tmp_dir, 'mapping.yaml'), 'w') as f:
        f.write(mapping_yaml)

    # Write the arch path so the worker subprocess can find it
    arch_path = (hw_config.arch_path
                 if hw_config and hw_config.arch_path
                 else None)

    meta = {
        'arch_path': arch_path,
    }
    with open(os.path.join(tmp_dir, 'eval_meta.yaml'), 'w') as f:
        yaml.dump(meta, f)

    worker_cmd = [sys.executable, '-m', 'polyfuse._looptree_worker', tmp_dir]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{project_root}:" + env.get('PYTHONPATH', '')

    try:
        result = subprocess.run(
            worker_cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
            cwd=project_root,
        )

        for line in result.stdout.splitlines():
            if line.startswith('LOOPTREE_ENERGY:'):
                return float(line.split(':', 1)[1])

        if result.returncode != 0:
            print(f"[LoopTree Error] Worker failed with rc={result.returncode}:\n{result.stderr}")
        else:
            print(f"[LoopTree Error] Worker failed:\n{result.stderr}")

        return float('inf')

    except subprocess.TimeoutExpired:
        print(f"[LoopTree Error] Worker timed out for block {layer_indices}")
        return float('inf')
    except Exception as e:
        print(f"[LoopTree Error] Exception launching worker: {e}")
        return float('inf')


def run_evaluation_on_partition(partition, all_layers, tmp_dir, hw_config=None):
    """
    Evaluate a full partition (list of blocks) using LoopTree.
    Each block is either a dict (from PolyFuse DP) or a list of indices.
    """
    total_energy = 0.0
    layer_name_to_idx = {l.get('name', f'layer{i}'): i
                         for i, l in enumerate(all_layers)}

    for block in partition:
        if isinstance(block, dict):
            layer_indices = [layer_name_to_idx[n]
                             for n in block['stack']
                             if n in layer_name_to_idx]
            tile_sizes = block.get('tile_sizes', {})
        else:
            layer_indices = block
            tile_sizes = {}

        layers = [all_layers[i] for i in layer_indices]
        block_energy = evaluate_single_block(
            layers, layer_indices, tile_sizes, tmp_dir, hw_config)
        total_energy += block_energy

    return total_energy
