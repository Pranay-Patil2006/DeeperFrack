path = '/deeper/PolyFuseV2/polyfuse/timeloop_integration.py'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'res = subprocess.run(cmd' in line and not 'print' in line:
        indent = len(line) - len(line.lstrip())
        new_lines.append(' ' * indent + 'print("Running cmd:", cmd)\n')
    new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
