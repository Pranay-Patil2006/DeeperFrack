import yaml

def modify_constraints(filepath, c_val, m_val, output_path):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
        
    for target in data['mapspace_constraints']['targets']:
        if target.get('type') == 'spatial':
            if target.get('target') == 'PEAccuBuffer':
                target['factors'] = f"P=1 Q=1 R=1 S=1 N=1 M=1 C={min(c_val, 8)}"
            elif target.get('target') == 'PEInputBuffer':
                target['factors'] = f"P=1 Q=1 R=1 S=1 N=1 C=1 M={min(m_val, 16)}"
                
    with open(output_path, 'w') as f:
        yaml.dump(data, f)

modify_constraints('/app/Examples/AlexNet_Simba/simba_like/constraints/simba_like_map_constraints.yaml', 3, 96, '/deeper/PolyFuseV2/data/components/dynamic_simba_like_map_constraints.yaml')
