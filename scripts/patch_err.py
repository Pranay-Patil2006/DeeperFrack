import re

path = '/deeper/PolyFuseV2/polyfuse/timeloop_integration.py'
with open(path, 'r') as f:
    content = f.read()

content = content.replace('raise RuntimeError("Timeloop failed or no valid mapping found")', 'raise RuntimeError(f"Timeloop failed: {res.stderr.decode()}")')
with open(path, 'w') as f:
    f.write(content)
