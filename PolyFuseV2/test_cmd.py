import subprocess
import os

timeloop_bin = '/opt/timeloop/bin/timeloop-mapper'
arch_path = '/app/Examples/AlexNet_Simba/simba_like/arch/simba_like.yaml'
layer_dir = '/deeper/PolyFuseV2/output/naive_baseline/layer_1'
prob_path = os.path.join(layer_dir, 'prob.yaml')
constraints_out = os.path.join(layer_dir, 'map_constraints.yaml')
components_dir = '/deeper/PolyFuseV2/data/components'

cmd = [timeloop_bin, arch_path, prob_path, constraints_out]
for f in os.listdir(components_dir):
    if f.endswith('.yaml') and f != 'simba_like_map_constraints.yaml' and not f.startswith('dynamic_'):
        cmd.append(os.path.join(components_dir, f))

print("CMD:", cmd)
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = '/opt/timeloop/lib:' + env.get('LD_LIBRARY_PATH', '')

res = subprocess.run(cmd, cwd=layer_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
print("RC:", res.returncode)
print("STDERR:", res.stderr.decode())
