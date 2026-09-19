path = '/deeper/PolyFuseV2/polyfuse/timeloop_integration.py'
with open(path, 'r') as f:
    content = f.read()

content = content.replace('logging.warning(f"PolyFuse Default Warning: Timeloop failed for Naive Layer {layer[\'name\']}, defaulting to analytical 1.5x MAC energy proxy!")', 'print(f"Error in Naive {layer[\'name\']}: {e}")\n                logging.warning(f"PolyFuse Default Warning: Timeloop failed for Naive Layer {layer[\'name\']}, defaulting to analytical 1.5x MAC energy proxy!")')
with open(path, 'w') as f:
    f.write(content)
