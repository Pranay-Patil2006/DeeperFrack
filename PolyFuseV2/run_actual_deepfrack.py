import sys
import os
import tempfile
from polyfuse.hal import HAL, NetworkLoader
from polyfuse.stage1_tiler import Stage1Tiler
from polyfuse.looptree_runner import run_evaluation_on_partition

arch = "/app/Examples/AlexNet_Simba/simba_like/arch/simba_like.yaml"
components = "data/components/"
network = "data/alexnet/"

hal = HAL(arch, components)
hw_config = hal.parse_all()
loader = NetworkLoader(network)
layers = loader.load()

tiler = Stage1Tiler(hw_config)

# DeepFrack partition:
# Stack 0: layer 0, 1
# Stack 1: layer 2
# Stack 2: layer 3
df_partition = [
    [layers[0], layers[1]],
    [layers[2]],
    [layers[3]]
]

final_partition = []
for stack in df_partition:
    is_valid, ts, wp, _ = tiler.evaluate_stack(stack)
    if not is_valid:
        print(f"Stack {stack} is invalid according to Tiler!")
    final_partition.append({
        'stack': [l['name'] for l in stack],
        'tile_sizes': ts
    })

print("Actual DeepFrack Partition:", final_partition)

with tempfile.TemporaryDirectory() as tmp_dir:
    energy = run_evaluation_on_partition(final_partition, layers, tmp_dir, hw_config)
    print(f"DeepFrack True LoopTree Energy: {energy} pJ")

