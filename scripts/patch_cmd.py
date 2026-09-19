path = '/deeper/PolyFuseV2/polyfuse/timeloop_integration.py'
with open(path, 'r') as f:
    content = f.read()

content = content.replace('res = subprocess.run(cmd', 'print("Running cmd:", cmd)\n            res = subprocess.run(cmd')
with open(path, 'w') as f:
    f.write(content)
