import yaml
import sys

def create_dynamic_constraints(c_val, m_val, output_path):
    data = {
        "mapspace_constraints": {
            "targets": [
                {
                    "target": "GlobalBuffer",
                    "type": "temporal",
                    "factors": "R=1 S=1",
                    "permutation": "RSCMN"
                },
                {
                    "target": "DRAM",
                    "type": "temporal",
                    "factors": "R=1 S=1",
                    "permutation": "RSCMN"
                },
                {
                    "target": "GlobalBuffer",
                    "type": "spatial",
                    "factors": f"P=1 Q=1 R=1 S=1 N=1 C=1 M={min(m_val, 16)}",
                    "permutation": "PQRSCNM"
                },
                {
                    "target": "PEAccuBuffer",
                    "type": "spatial",
                    "factors": f"P=1 Q=1 R=1 S=1 N=1 M=1 C={min(c_val, 8)}",
                    "permutation": "PQRSCNM"
                },
                {
                    "target": "PEWeightRegs",
                    "type": "temporal",
                    "factors": "R=1 S=1 C=1 M=1 N=1",
                    "permutation": "RSCMN"
                },
                {
                    "target": "PEInputBuffer",
                    "type": "temporal",
                    "factors": "P=1 Q=1 R=1 S=1 C=1 M=1 N=1",
                    "permutation": "PQRSCMN"
                },
                {
                    "target": "PEAccuBuffer",
                    "type": "temporal",
                    "factors": "P=1 Q=1 R=1 S=1 C=1 N=1",
                    "permutation": "PQRSCN"
                },
                {
                    "target": "PEWeightBuffer",
                    "type": "temporal",
                    "factors": "P=1 Q=1 M=1 N=1",
                    "permutation": "PQMN"
                }
            ]
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

create_dynamic_constraints(3, 96, '/deeper/PolyFuseV2/data/components/dynamic_simba_like_map_constraints.yaml')
