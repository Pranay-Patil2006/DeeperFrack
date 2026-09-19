import argparse
import sys
import os
import tempfile
from hal import HAL, NetworkLoader
from stage1_tiler import Stage1Tiler
import pprint

hardware_object = HAL("/app/Examples/AlexNet_Simba/simba_like/arch/simba_like.yaml", "/app/Examples/AlexNet_Simba/simba_like/arch/components")
hardware_config = hardware_object.parse_all()
network = NetworkLoader("/deeper/PolyFuseV2/data/alexnet")
network.load()
tiler  = Stage1Tiler(hardware_config) 

pprint.pprint(hardware_config.memory_hierarchy)
print("Type : ", type(hardware_config))
print("Object : ",hardware_config)
print()

print("cap_dram : ", tiler.cap_dram)
print("cap_global : ", tiler.cap_global)
print("cap_pe_input : ", tiler.cap_pe_input)
print("cap_pe_weight : ", tiler.cap_pe_weight)
print("cap_pe_output : ", tiler.cap_pe_output)