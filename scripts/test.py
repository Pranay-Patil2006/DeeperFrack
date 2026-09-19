import sys
import os

path = '/deeper/PolyFuseV2/polyfuse/timeloop_integration.py'
with open(path, 'r') as f:
    content = f.read()

content = content.replace('res = subprocess.run(cmd, cwd=layer_dir', 'print("CMD:", cmd)\n                res = subprocess.run(cmd, cwd=layer_dir')
with open(path, 'w') as f:
    f.write(content)
