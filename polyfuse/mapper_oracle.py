import os
import subprocess
import yaml
import tempfile

def get_optimal_tile_sizes(layer_file, arch_file, constraints_file, components_dir, mapper_file):
    """
    Runs timeloop-mapper on a single layer and parses the output map.yaml 
    to extract the absolute best tile sizes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build component glob
        comp_glob = os.path.join(components_dir, '*.yaml')
        
        # Run timeloop-mapper
        cmd = [
            "/tmp/accelergy-timeloop-infrastructure/src/timeloop/build/timeloop-mapper",
            arch_file,
            layer_file,
            mapper_file,
            constraints_file,
            comp_glob,
            "-o", tmpdir
        ]
        
        try:
            # We run it in a shell to expand the glob
            subprocess.run(" ".join(cmd), shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running timeloop-mapper for {layer_file}: {e.stderr}")
            return {}
            
        map_yaml_path = os.path.join(tmpdir, "timeloop-mapper.map.yaml")
        if not os.path.exists(map_yaml_path):
            return {}
            
        with open(map_yaml_path, 'r') as f:
            mapping_data = yaml.safe_load(f)
            
        # Parse factors
        # We look for temporal loops
        tile_sizes = {'T_out': 1, 'T_in': 1, 'T_c': 1, 'T_m': 1, 'T_r': 1, 'T_s': 1}
        
        # In Timeloop, a rank can appear at multiple levels. We need the product of factors
        # for spatial and temporal across all levels below DRAM?
        # Actually, if we just want the tile sizes for the LoopTree compute node, we need to
        # know the size of the block that fits in the GlobalBuffer (L2) or L1.
        # LoopTree expects the tile_shape to be the size of the loop.
        
        # Wait, if we use LoopTree, we just need the loop bounds.
        # Let's accumulate the factors for each dimension across all levels except the outermost?
        pass

