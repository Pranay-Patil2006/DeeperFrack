"""
Standalone LoopTree worker (subprocess isolation).

Reads:
  <tmp_dir>/workload.yaml   — fused workload definition
  <tmp_dir>/mapping.yaml    — symbolic LoopTree mapping
  <tmp_dir>/eval_meta.yaml  — metadata: {arch_path: <path to real arch yaml>}

Generates a pytimeloop v0.4 arch YAML that:
  - Uses the real Simba component subclasses (smartbuffer_SRAM, lmac)
    so Accelergy looks up the correct energy tables
  - Is in the flat v0.4 nodes format that run_looptree requires
    (v0.3 subtree/local format is NOT supported by pytimeloop)

Prints:
  LOOPTREE_ENERGY:<float>   — total energy in pJ
"""
import sys
import os
import yaml
import shutil
import subprocess

sys.path.insert(0, "/tmp/accelergy-timeloop-infrastructure/src/timeloop-python")

from pytimeloop.looptree.run import run_looptree
from pathlib import Path


# pytimeloop target bindings: maps integer target ID -> component name
# Must match names in the arch YAML we generate below AND the mapping YAML.
BINDINGS = {
    0: 'DRAM',
    1: 'GlobalBuffer',
    2: 'PEAccuBuffer',
    3: 'PEWeightBuffer',
    4: 'PEInputBuffer',
    5: 'LMAC',
}

# v0.4 Simba-compatible arch YAML.

def parse_arch_v03_to_v04(arch_dict):
    """
    Dynamically converts a v0.3 tree architecture into a v0.4 flat nodes list
    so we don't 'make up' a hierarchy but accurately reflect the user's YAML.
    """
    components = []
    
    def dfs(node):
        if isinstance(node, dict):
            if 'local' in node:
                for comp in node['local']:
                    name = comp['name'].split('[')[0]
                    c_class = comp.get('subclass', comp.get('class', 'storage'))
                    c = {
                        'name': name,
                        'class': c_class,
                        'attributes': comp.get('attributes', {})
                    }
                    if c_class.lower() in ('lmac', 'mac', 'compute'):
                        c['required_actions'] = ['compute', 'leak']
                    else:
                        c['required_actions'] = ['read', 'write', 'update']
                    components.append(c)
            if 'subtree' in node:
                for child in node['subtree']:
                    dfs(child)
        elif isinstance(node, list):
            for item in node:
                dfs(item)
                
    if 'architecture' in arch_dict:
        dfs(arch_dict['architecture'].get('subtree', []))
    else:
        dfs(arch_dict.get('subtree', []))
        
    return components

def main():
    if len(sys.argv) < 2:
        print("ERROR: missing tmp_dir argument", file=sys.stderr)
        sys.exit(1)

    tmp_dir = sys.argv[1]

    # Read metadata to find the real arch file (for locating components/)
    meta_path = os.path.join(tmp_dir, 'eval_meta.yaml')
    arch_path = None
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = yaml.safe_load(f)
        arch_path = meta.get('arch_path')

    # --- Step 1: Generate ERT using Accelergy CLI with the ORIGINAL v0.3 arch ---
    ert_path = os.path.join(tmp_dir, 'ERT.yaml')

    if arch_path and os.path.exists(arch_path):
        comp_dir = os.path.join(os.path.dirname(arch_path), 'components')
        comp_files = (
            [os.path.join(comp_dir, f)
             for f in os.listdir(comp_dir) if f.endswith('.yaml')]
            if os.path.isdir(comp_dir) else []
        )

        accel_cmd = ['accelergy', '-v', '-o', tmp_dir, arch_path] + comp_files
        result = subprocess.run(accel_cmd, capture_output=True, text=True)

        if result.returncode != 0 or not os.path.exists(ert_path):
            print(f"LOOPTREE_ERROR: Accelergy failed: {result.stderr[-200:]}",
                  file=sys.stderr)
            sys.exit(2)
    else:
        print("LOOPTREE_ERROR: No arch_path for Accelergy", file=sys.stderr)
        sys.exit(2)

    # Parse the real architecture to build the v0.4 equivalent
    with open(arch_path, 'r') as f:
        arch_dict = yaml.safe_load(f)
        
    v04_components = parse_arch_v03_to_v04(arch_dict)
    
    v04_yaml_lines = ["architecture:", "  version: 0.4", "  nodes:"]
    valid_names = set()
    dynamic_bindings = {}
    for idx, c in enumerate(v04_components):
        valid_names.add(c['name'])
        dynamic_bindings[idx] = c['name']
        v04_yaml_lines.append(f"  - name: {c['name']}")
        v04_yaml_lines.append(f"    class: {c['class']}")
        if c['attributes'] or True:
            v04_yaml_lines.append(f"    attributes:")
            clean_attrs = {}
            for k, v in c['attributes'].items():
                k_clean = k.replace('-', '_')
                if k_clean == 'memory_depth': k_clean = 'depth'
                elif k_clean == 'memory_width': k_clean = 'width'
                elif k_clean in ('nports', 'num_ports', 'n_rdwr_ports', 'n_rd_ports', 'n_wr_ports'):
                    continue
                clean_attrs[k_clean] = v

            if 'datawidth' not in clean_attrs:
                clean_attrs['datawidth'] = 16
            
            if 'smartbuffer' in c['class'].lower() or 'sram' in c['class'].lower():
                if 'n_rdwr_ports' not in clean_attrs and 'n_rd_ports' not in clean_attrs:
                    clean_attrs['n_rdwr_ports'] = 2
                    clean_attrs['n_rd_ports'] = 0
                    clean_attrs['n_wr_ports'] = 0

            # timeloopfe crashes on missing depth for any storage, even DRAM
            if c['class'].lower() != 'compute' and 'mac' not in c['class'].lower():
                if 'depth' not in clean_attrs:
                    clean_attrs['depth'] = 0

            for k, v in clean_attrs.items():
                v04_yaml_lines.append(f"      {k}: {v}")
        v04_yaml_lines.append(f"    required_actions: {c['required_actions']}")
        
    arch_local = os.path.join(tmp_dir, 'arch.yaml')
    with open(arch_local, 'w') as f:
        f.write("\n".join(v04_yaml_lines))

    # --- Step 1b: Rewrite ERT.yaml to match v0.4 flat component names dynamically ---
    with open(ert_path, 'r') as f:
        ert = yaml.safe_load(f)

    new_tables = []
    has_dram = False
    for t in ert['ERT']['tables']:
        # Match ERT names to the dynamically extracted component names
        for vname in valid_names:
            if vname in t['name']:
                t['name'] = vname
                new_tables.append(t)
                if vname == 'DRAM':
                    has_dram = True
                break

    if not has_dram and 'DRAM' in valid_names:
        new_tables.append({
            'name': 'DRAM',
            'actions': [
                {'name': 'read', 'energy': 512.0, 'arguments': {}},
                {'name': 'write', 'energy': 512.0, 'arguments': {}},
                {'name': 'update', 'energy': 512.0, 'arguments': {}}
            ]
        })

    ert['ERT']['tables'] = new_tables
    with open(ert_path, 'w') as f:
        yaml.dump(ert, f)

    # --- Step 3: Run LoopTree with v0.4 arch + pre-generated ERT ---
    from pytimeloop.looptree.run import run_looptree
    
    try:
        stats = run_looptree(
            Path(tmp_dir),
            ['workload.yaml', 'mapping.yaml', 'arch.yaml', 'ERT.yaml'],
            tmp_dir,
            dynamic_bindings,
            False,   # ERT already generated above
        )

        if isinstance(stats.energy, dict):
            total = sum(v for v in stats.energy.values()
                        if isinstance(v, (int, float)))
        else:
            total = float(stats.energy)

        print(f"LOOPTREE_ENERGY:{total}")

    except RuntimeError as e:
        print(f"LOOPTREE_ERROR:{e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(f"LOOPTREE_ERROR:{e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
