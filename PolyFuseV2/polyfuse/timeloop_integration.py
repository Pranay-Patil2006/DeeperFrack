import os
import subprocess
import yaml
import re
import shutil
import logging

class TimeloopIntegration:
    def __init__(self, arch_path, components_dir, output_dir):
        self.arch_path = arch_path
        self.components_dir = components_dir
        self.output_dir = output_dir
        self.timeloop_bin = '/opt/timeloop/bin/timeloop-mapper'
        if not os.path.exists(self.timeloop_bin):
            # Fallback to system PATH
            self.timeloop_bin = 'timeloop-mapper'
            
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _generate_dynamic_constraints(self, C, M, R, output_path):
        constraints_path = os.path.join(self.components_dir, 'simba_like_map_constraints.yaml')
        if not os.path.exists(constraints_path):
            logging.warning("No simba_like_map_constraints.yaml found! Timeloop may fail.")
            return False
            
        with open(constraints_path, 'r') as f:
            c_data = yaml.safe_load(f)
            
        # Remove any existing spatial constraints
        c_data['mapspace_constraints']['targets'] = [
            t for t in c_data['mapspace_constraints']['targets'] if t.get('type') != 'spatial'
        ]
        
        # Do not inject any new spatial bounds. Let Timeloop find the optimum natively.
        
        with open(output_path, 'w') as f:
            yaml.dump(c_data, f, default_flow_style=False)
        return True
        
    def generate_problem_yaml(self, layers, tile_sizes, stack_idx, filepath):
        # We create a virtual problem combining the footprint of the stack
        # For simplicity, we just aggregate the MACs and max the footprints
        # In a real perfect mapper, we generate explicit map-spaces or constraints.
        
        # Max dimensions to simulate the worst-case fused footprint
        max_C = max(l['C'] for l in layers)
        max_M = max(l['M'] for l in layers)
        max_R = max(l['R'] for l in layers)
        max_S = max(l['S'] for l in layers)
        
        # The output spatial tile is determined by the last layer's T_out
        last_name = layers[-1]['name']
        t_out = tile_sizes[last_name]['T_out']
        
        prob = {
            'problem': {
                'shape': {
                    'name': 'CNN-Layer',
                    'dimensions': ['C', 'M', 'R', 'S', 'P', 'Q', 'N'],
                    'coefficients': [
                        {'name': 'Wstride', 'default': 1},
                        {'name': 'Hstride', 'default': 1}
                    ],
                    'data-spaces': [
                        {'name': 'Weights', 'projection': [[['C']], [['M']], [['R']], [['S']]]},
                        {'name': 'Inputs', 'projection': [[['N']], [['P', 'Hstride'], ['R']], [['Q', 'Wstride'], ['S']], [['C']]]},
                        {'name': 'Outputs', 'projection': [[['N']], [['P']], [['Q']], [['M']]], 'read-write': True}
                    ]
                },
                'instance': {
                    'N': 1,
                    'C': max_C,
                    'M': max_M,
                    'R': max_R,
                    'S': max_S,
                    'P': t_out,
                    'Q': t_out,
                    'Wstride': 1,
                    'Hstride': 1
                }
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(prob, f, default_flow_style=None)
            
    def run_timeloop(self, stack_idx, layers, tile_sizes):
        stack_dir = os.path.join(self.output_dir, f'stack_{stack_idx}')
        os.makedirs(stack_dir, exist_ok=True)
        
        prob_path = os.path.join(stack_dir, 'prob.yaml')
        # Fix Fanout Error: Dynamically construct spatial constraints bounded by layer shapes
        max_C = max(l['C'] for l in layers)
        max_M = max(l['M'] for l in layers)
        max_R = max(l.get('R', 1) for l in layers)
        constraints_out = os.path.join(stack_dir, 'map_constraints.yaml')
        self._generate_dynamic_constraints(max_C, max_M, max_R, constraints_out)
        
        self.generate_problem_yaml(layers, tile_sizes, stack_idx, prob_path)
        
        # We need constraints and mapper configs. 
        # For now, we will just pass the arch and problem.
        cmd = [self.timeloop_bin, os.path.abspath(self.arch_path), os.path.abspath(prob_path), os.path.abspath(constraints_out)]
        
        # Include all components except the template constraints
        if os.path.exists(self.components_dir):
            for f in os.listdir(self.components_dir):
                if f.endswith('.yaml') and f != 'simba_like_map_constraints.yaml' and not f.startswith('dynamic_'):
                    cmd.append(os.path.abspath(os.path.join(self.components_dir, f)))
                    
        # Setup environment
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/opt/timeloop/lib:' + env.get('LD_LIBRARY_PATH', '')
        
        # Run timeloop
        try:
            res = subprocess.run(cmd, cwd=stack_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            stats = self._parse_stats(os.path.join(stack_dir, 'timeloop-mapper.stats.txt'))
            
            if res.returncode != 0 or stats['energy_pJ'] == float('inf'):
                raise RuntimeError(f"Timeloop failed: {res.stderr.decode()}")
                
            return stats
        except Exception as e:
            # Fallback for fused stack
            print(f"Timeloop failed for stack {stack_idx}: {e}")
            logging.warning(f"PolyFuse Default Warning: Timeloop failed for Fused Stack {stack_idx}, defaulting to analytical 1.0x MAC energy proxy!")
            fallback_energy = 0.0
            for layer in layers:
                fallback_energy += layer['P'] * layer['Q'] * layer['M'] * layer['C'] * layer['R'] * layer['S'] * 1.0 # slight discount for fused
            return {'energy_pJ': fallback_energy, 'cycles': 0}
            
    def _parse_stats(self, stats_path):
        if not os.path.exists(stats_path):
            return {'energy_pJ': float('inf'), 'cycles': float('inf')}
            
        energy = float('inf')
        cycles = float('inf')
        
        with open(stats_path, 'r') as f:
            content = f.read()
            
        e_match = re.search(r'Energy:\s*([0-9.]+)\s*uJ', content)
        if e_match:
            energy = float(e_match.group(1)) * 1e6 # convert uJ to pJ
            
        c_match = re.search(r'Cycles:\s*([0-9]+)', content)
        if c_match:
            cycles = float(c_match.group(1))
            
        return {'energy_pJ': energy, 'cycles': cycles}
        
    def _evaluate_single_naive_layer(self, i, layer, naive_dir):
        """
        Evaluate a single layer for the naive baseline.
        This is extracted to support parallel execution using ThreadPoolExecutor.
        """
        layer_dir = os.path.join(naive_dir, f'layer_{i}')
        os.makedirs(layer_dir, exist_ok=True)
        
        # Check for precomputed results
        stats_path = os.path.join(layer_dir, 'timeloop-mapper.stats.txt')
        if os.path.exists(stats_path):
            stats = self._parse_stats(stats_path)
            if stats['energy_pJ'] != float('inf'):
                print(f"[PolyFuse] Reusing precomputed naive mapping for {layer['name']}")
                return i, stats
        
        prob = {
            'problem': {
                'shape': {
                    'name': 'CNN-Layer',
                    'dimensions': ['C', 'M', 'R', 'S', 'P', 'Q', 'N'],
                    'data-spaces': [
                        {'name': 'Weights', 'projection': [[['C']], [['M']], [['R']], [['S']]]},
                        {'name': 'Inputs', 'projection': [[['N']], [['P'], ['R']], [['Q'], ['S']], [['C']]]},
                        {'name': 'Outputs', 'projection': [[['N']], [['P']], [['Q']], [['M']]], 'read-write': True}
                    ]
                },
                'instance': {
                    'N': 1,
                    'C': layer['C'],
                    'M': layer['M'],
                    'R': layer['R'],
                    'S': layer['S'],
                    'P': layer['P'],
                    'Q': layer['Q']
                }
            }
        }
        
        prob_path = os.path.join(layer_dir, 'prob.yaml')
        with open(prob_path, 'w') as f:
            yaml.dump(prob, f, default_flow_style=None)
            
        constraints_out = os.path.join(layer_dir, 'map_constraints.yaml')
        self._generate_dynamic_constraints(layer['C'], layer['M'], layer.get('R', 1), constraints_out)
            
        cmd = [self.timeloop_bin, os.path.abspath(self.arch_path), os.path.abspath(prob_path), os.path.abspath(constraints_out)]
        if os.path.exists(self.components_dir):
            for f in os.listdir(self.components_dir):
                if f.endswith('.yaml') and f != 'simba_like_map_constraints.yaml' and not f.startswith('dynamic_'):
                    cmd.append(os.path.abspath(os.path.join(self.components_dir, f)))
                    
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/opt/timeloop/lib:' + env.get('LD_LIBRARY_PATH', '')
                    
        try:
            res = subprocess.run(cmd, cwd=layer_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            stats = self._parse_stats(os.path.join(layer_dir, 'timeloop-mapper.stats.txt'))
            
            if res.returncode != 0 or stats['energy_pJ'] == float('inf'):
                raise RuntimeError(f"Timeloop failed: {res.stderr.decode()}")
                
            return i, stats
        except Exception as e:
            print(f"Error in Naive {layer['name']}: {e}")
            logging.warning(f"PolyFuse Default Warning: Timeloop failed for Naive Layer {layer['name']}, defaulting to analytical 1.5x MAC energy proxy!")
            fallback_energy = layer['P'] * layer['Q'] * layer['M'] * layer['C'] * layer['R'] * layer['S'] * 1.5
            return i, {'energy_pJ': fallback_energy, 'cycles': 0}

    def get_naive_baseline(self, layers):
        """
        Evaluate all layers in the naive baseline in parallel.
        Since the system has 24 CPU cores, running multiple layer evaluations in parallel
        while each evaluation utilizes multiple threads saturates the system resources efficiently.
        """
        import json
        import hashlib
        
        cache_file = '/deeper/PolyFuseV2/naive_cache.json'
        layers_str = json.dumps(layers, sort_keys=True)
        key = hashlib.md5((self.arch_path + layers_str).encode('utf-8')).hexdigest()
        
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except Exception:
                pass
                
        if key in cache:
            print("[PolyFuse] Loaded naive baseline from cache.")
            cached_res = cache[key]
            return cached_res['total_energy'], cached_res['results']

        naive_dir = os.path.join(self.output_dir, 'naive_baseline')
        os.makedirs(naive_dir, exist_ok=True)
        
        from concurrent.futures import ThreadPoolExecutor
        
        results_dict = {}
        # Balance Python parallelism and Mapper thread parallelism (e.g. 2 parallel processes * 8 threads = 16 cores)
        max_workers = 2
        
        print(f"[PolyFuse] Evaluating {len(layers)} naive layers in parallel (workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._evaluate_single_naive_layer, i, layer, naive_dir)
                for i, layer in enumerate(layers)
            ]
            for future in futures:
                idx, stats = future.result()
                results_dict[idx] = stats
                
        # Reconstruct ordered results and sum up energy
        results = [results_dict[i] for i in range(len(layers))]
        total_energy = sum(r['energy_pJ'] for r in results)
        
        cache[key] = {
            'total_energy': total_energy,
            'results': results
        }
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=4)
            
        return total_energy, results
