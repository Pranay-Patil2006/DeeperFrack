import json
import os
import yaml

class Reporter:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_report(self, partition, naive_energy, naive_results, fused_results):
        total_fused_energy = sum(res['energy_pJ'] for res in fused_results)
        
        reduction = 0.0
        if naive_energy > 0:
            reduction = (naive_energy - total_fused_energy) / naive_energy * 100.0
            
        md_content = "# PolyFuse V2 Final Report\n\n"
        md_content += "## Verification Results\n\n"
        md_content += "| Metric | Naive (SLC) | PolyFuse V2 (Fused) | Reduction |\n"
        md_content += "|:---|---:|---:|---:|\n"
        
        # Naive breakdown
        for i, res in enumerate(naive_results):
            md_content += f"| Layer {i} Energy | {res['energy_pJ']:.2f} pJ | - | - |\n"
            
        # Fused breakdown
        for i, stack in enumerate(partition):
            stack_layers = ", ".join(stack['stack'])
            md_content += f"| Stack {i} Fused ({stack_layers}) | - | {fused_results[i]['energy_pJ']:.2f} pJ | - |\n"
            
        md_content += f"| **Total** | **{naive_energy:.2f} pJ** | **{total_fused_energy:.2f} pJ** | **{reduction:.2f}%** |\n\n"
        
        md_content += "## Partition Details\n\n"
        for i, stack in enumerate(partition):
            md_content += f"### Stack {i}\n"
            md_content += f"- **Layers:** {', '.join(stack['stack'])}\n"
            md_content += f"- **Weight Caching Pattern:** {', '.join(stack['weight_pattern']) if stack['weight_pattern'] else 'None'}\n"
            md_content += "- **Tile Sizes:**\n"
            for lname, ts in stack['tile_sizes'].items():
                md_content += f"  - {lname}: T_in={ts['T_in']}, T_out={ts['T_out']}\n"
            md_content += "- **Spatial Routing:**\n"
            for idx, r in enumerate(stack['routing']):
                md_content += f"  - Layer {idx}: X={r[0]}, Y={r[1]}\n"
            md_content += "\n"
            
        with open(os.path.join(self.output_dir, 'report.md'), 'w') as f:
            f.write(md_content)
            
        # Write results.json
        res_json = {
            'naive_total_energy_pJ': naive_energy,
            'fused_total_energy_pJ': total_fused_energy,
            'reduction_percentage': reduction,
            'partition': partition,
            'fused_results': fused_results
        }
        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(res_json, f, indent=2)
            
        # Write schedule.yaml (abstract representation for now)
        schedule = {'schedule': partition}
        with open(os.path.join(self.output_dir, 'schedule.yaml'), 'w') as f:
            yaml.dump(schedule, f)
            
        return md_content
