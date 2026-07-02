import yaml

def modify_constraints(filepath, output_path):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
        
    targets = data['mapspace_constraints']['targets']
    # Remove all spatial targets
    targets = [t for t in targets if t.get('type') != 'spatial']
    
    # Add our own spatial targets
    targets.append({
        'target': 'GlobalBuffer',
        'type': 'spatial',
        'factors': 'M=16 P=1 Q=1 R=1 S=1 N=1 C=1',
        'permutation': 'MPQRSCN'
    })
    # Add a spatial target for the MACs (assuming PEWeightRegs or PEAccuBuffer)
    # Let's just try GlobalBuffer first and see what other fanouts exist
    
    data['mapspace_constraints']['targets'] = targets
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f)

modify_constraints('/app/Examples/AlexNet_Simba/simba_like/constraints/simba_like_map_constraints.yaml', '/deeper/PolyFuseV2/data/components/dynamic_simba_like_map_constraints.yaml')
