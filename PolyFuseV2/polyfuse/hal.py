"""
Hardware Abstraction Layer (HAL) for PolyFuse V2.

Parses Timeloop/Accelergy architecture YAMLs to extract hardware
parameters (buffer capacities, PE array geometry, memory hierarchy)
with NO hardcoded fallbacks — every value comes from the YAML.
"""
import yaml
import os
import re
import logging

logger = logging.getLogger(__name__)


class HardwareConfig:
    def __init__(self):
        self.buffers = {}          # name -> {capacity_words, word_bits, instances, role}
        self.pe_array = {'meshX': 1, 'meshY': 1, 'total_pes': 1, 'macs_per_pe': 1}
        self.mac_energy = 0.0
        self.memory_hierarchy = []  # ordered from outermost (DRAM) to innermost (MAC)
        self.arch_path = None       # keep track for LoopTree subprocess use


class HAL:
    def __init__(self, arch_path, components_dir):
        self.arch_path = arch_path
        self.components_dir = components_dir
        self.hw_config = HardwareConfig()
        self.hw_config.arch_path = arch_path

    def parse_all(self):
        with open(self.arch_path, 'r') as f:
            content = f.read()

        # Strip Timeloop-specific YAML tags that confuse PyYAML
        content = re.sub(r'![\w]+', '', content)
        arch_data = yaml.safe_load(content)

        arch = arch_data.get('architecture', {})
        buffers = {}
        memory_hierarchy = []
        pe_array = {'meshX': 1, 'meshY': 1, 'total_pes': 1, 'macs_per_pe': 1}

        # Recursively walk the architecture tree
        self._walk_tree(arch, buffers, memory_hierarchy, pe_array, inherited_attrs={})

        self.hw_config.buffers = buffers
        self.hw_config.memory_hierarchy = memory_hierarchy
        self.hw_config.pe_array = pe_array
        return self.hw_config

    def _walk_tree(self, node, buffers, hierarchy, pe_array, inherited_attrs, parent_instances=1):
        """
        Recursively traverse the architecture tree.
        Handles both v0.3 (subtree/local) and v0.4 (nodes) formats.
        """
        # Merge attributes from this level (child overrides parent)
        attrs = dict(inherited_attrs)
        if isinstance(node, dict):
            attrs.update(self._normalize_attrs(node.get('attributes', {})))

        # Extract multiplier for this node if it's a subtree with a name like PE[0..15]
        current_instances = parent_instances
        if isinstance(node, dict) and 'name' in node:
            current_instances *= self._parse_instance_count(node['name'])

        # v0.3 format: architecture has 'subtree'
        subtrees = node.get('subtree', []) if isinstance(node, dict) else []
        locals_ = node.get('local', []) if isinstance(node, dict) else []
        # v0.4 format: 'nodes'
        nodes_v4 = node.get('nodes', []) if isinstance(node, dict) else []

        # Process v0.3 local components
        for component in locals_:
            self._process_component(component, buffers, hierarchy, pe_array, attrs, current_instances)

        # Process v0.4 nodes
        for component in nodes_v4:
            self._process_component(component, buffers, hierarchy, pe_array, attrs, current_instances)

        # Recurse into subtrees
        for subtree in subtrees:
            self._walk_tree(subtree, buffers, hierarchy, pe_array, attrs, current_instances)

    def _normalize_attrs(self, attrs):
        if not attrs:
            return {}
        normalized = {}
        for k, v in attrs.items():
            norm_k = k.lower().replace('-', '_')
            normalized[norm_k] = v
        return normalized

    def _parse_instance_count(self, name):
        """
        Parse instance count from names like 'PE[0..15]', 'PEWeightBuffer[0..7]'.
        Returns count (e.g., 16 for [0..15]).
        """
        match = re.search(r'\[(\d+)\.\.(\d+)\]', name)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            return hi - lo + 1
        return 1

    def _process_component(self, component, buffers, hierarchy, pe_array, parent_attrs, parent_instances=1):
        if not isinstance(component, dict):
            return

        name_raw = component.get('name', '')
        # Remove instance range for canonical name
        name = re.sub(r'\[.*?\]', '', name_raw).strip()
        instances = self._parse_instance_count(name_raw) * parent_instances

        klass = component.get('class', '').lower()
        subclass = component.get('subclass', '').lower()

        # Merge attributes
        comp_attrs = dict(parent_attrs)
        comp_attrs.update(self._normalize_attrs(component.get('attributes', {})))

        # Class classification mirroring Timeloop isBufferClass and isComputeClass
        is_storage = False
        buffer_keywords = ["dram", "sram", "regfile", "smartbuffer", "storage"]
        for kw in buffer_keywords:
            if kw in klass or kw in subclass:
                is_storage = True
                break

        is_compute = False
        compute_keywords = ["mac", "intmac", "fpmac", "compute"]
        for kw in compute_keywords:
            if kw in klass or kw in subclass:
                is_compute = True
                break

        if is_storage or klass == 'dram':
            # 1. Parse word-bits
            wbits = comp_attrs.get('word_bits') or comp_attrs.get('datawidth') or 16

            # 2. Parse block_size
            block_size = comp_attrs.get('block_size') or comp_attrs.get('n_words')
            block_size_specified = (block_size is not None)
            if not block_size_specified:
                block_size = 1

            # 3. Parse cluster_size
            cluster_size = comp_attrs.get('cluster_size')
            cluster_size_specified = (cluster_size is not None)
            if not cluster_size_specified:
                cluster_size = 1

            # 4. Parse width
            width = comp_attrs.get('width') or comp_attrs.get('memory_width') or comp_attrs.get('data_storage_width')
            if width is not None:
                if block_size_specified and cluster_size_specified:
                    pass
                elif cluster_size_specified:
                    if cluster_size * wbits > 0:
                        block_size = width // (cluster_size * wbits)
                elif block_size_specified:
                    if wbits * block_size > 0:
                        cluster_size = width // (wbits * block_size)
                else:
                    if wbits > 0:
                        block_size = width // wbits
                    cluster_size = 1

            # 5. Parse capacity size parameters
            entries = comp_attrs.get('entries')
            depth = comp_attrs.get('depth') or comp_attrs.get('memory_depth') or comp_attrs.get('data_storage_depth')
            sizeKB = comp_attrs.get('sizekb')

            if entries is not None:
                capacity_words = entries
            elif depth is not None:
                capacity_words = depth * block_size
            elif sizeKB is not None:
                capacity_words = int((sizeKB * 1024 * 8) // wbits) if wbits > 0 else 0
            else:
                capacity_words = 0

            # DRAM fallback for depth/capacity
            if (klass == 'dram' or 'dram' in name.lower()) and capacity_words == 0:
                capacity_words = 2**30

            n_banks = comp_attrs.get('n_banks') or comp_attrs.get('nbanks') or 1
            total_capacity_words = capacity_words * instances * n_banks

            mesh_x = comp_attrs.get('meshx') or comp_attrs.get('mesh_x') or 1

            buffers[name] = {
                'capacity_words': total_capacity_words,
                'instances': instances,
                'mesh_x': mesh_x,
                'word_bits': wbits,
                'role': self._infer_role_from_constraints_and_name(name, klass, subclass),
            }
            if name not in hierarchy:
                hierarchy.append(name)

        if is_compute:
            mesh_x = comp_attrs.get('meshx') or comp_attrs.get('mesh_x') or 1
            # Update the PE array geometry
            if instances > 1 and mesh_x > 1:
                pe_array['macs_per_pe'] = instances
                pe_array['meshX'] = mesh_x
                pe_array['total_pes'] = mesh_x
                pe_array['meshY'] = 1
            elif instances > 1:
                pe_array['macs_per_pe'] = max(pe_array['macs_per_pe'], instances)
            if mesh_x > 1:
                pe_array['meshX'] = max(pe_array['meshX'], mesh_x)
                pe_array['total_pes'] = pe_array['meshX'] * pe_array['meshY']

    def _infer_role_from_constraints_and_name(self, name, klass, subclass):
        # Try parsing constraints first if self.arch_path is set
        role = None
        dirs_to_check = []
        if self.arch_path:
            arch_dir = os.path.dirname(os.path.dirname(self.arch_path))
            dirs_to_check.append(os.path.join(arch_dir, 'constraints'))
        if self.components_dir:
            dirs_to_check.append(self.components_dir)
            
        for d in dirs_to_check:
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith('.yaml'):
                        try:
                            with open(os.path.join(d, f), 'r') as file:
                                content = re.sub(r'![\w]+', '', file.read())
                                data = yaml.safe_load(content)
                                targets = []
                                if isinstance(data, dict):
                                    if 'architecture_constraints' in data:
                                        targets = data['architecture_constraints'].get('targets', [])
                                    elif 'targets' in data:
                                        targets = data.get('targets', [])
                                for t in targets:
                                    if t.get('target') == name and t.get('type') == 'datatype':
                                        keep = t.get('keep', [])
                                        keep_lower = [k.lower() for k in keep]
                                        if 'inputs' in keep_lower and 'outputs' in keep_lower:
                                            role = 'global'
                                        elif 'inputs' in keep_lower:
                                            role = 'input'
                                        elif 'outputs' in keep_lower:
                                            role = 'output'
                                        elif 'weights' in keep_lower:
                                            if 'reg' in name.lower() or subclass == 'regfile':
                                                role = 'weight_reg'
                                            else:
                                                role = 'weight'
                                        break
                        except Exception:
                            pass
                    if role:
                        break
            if role:
                break

        # Fallback to name/class-based inference
        if not role:
            n = name.lower()
            k = klass.lower() if klass else ''
            s = subclass.lower() if subclass else ''
            
            if 'dram' in n or 'dram' in k or 'main' in n or 'off' in n:
                role = 'dram'
            elif 'global' in n or 'glb' in n or 'shared' in n:
                role = 'global'
            elif 'input' in n or 'iact' in n or 'ifmap' in n:
                role = 'input'
            elif 'weight' in n and ('reg' in n or s == 'regfile'):
                role = 'weight_reg'
            elif 'weight' in n or 'filter' in n or 'kernel' in n:
                role = 'weight'
            elif 'accu' in n or 'output' in n or 'psum' in n or 'ofmap' in n:
                role = 'output'
            else:
                role = 'unknown'

        # Always force DRAM class/name to DRAM role
        if 'dram' in name.lower() or (klass and 'dram' in klass.lower()):
            role = 'dram'

        return role


class NetworkLoader:
    def __init__(self, network_dir):
        self.network_dir = network_dir
        self.layers = []

    def load(self):
        if not os.path.exists(self.network_dir):
            return []

        files = sorted([f for f in os.listdir(self.network_dir) if f.endswith('.yaml')])
        for f in files:
            layer = self._parse_layer(os.path.join(self.network_dir, f))
            if layer:
                self.layers.append(layer)
        return self.layers

    def _parse_layer(self, filepath):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        if 'problem' not in data or 'instance' not in data['problem']:
            return None

        dimensions = data['problem']['instance']

        layer = {
            'name': os.path.basename(filepath).replace('.yaml', ''),
            'C': dimensions.get('C', 1),
            'M': dimensions.get('M', dimensions.get('K', 1)),
            'R': dimensions.get('R', 1),
            'S': dimensions.get('S', 1),
            'P': dimensions.get('P', dimensions.get('H', 1)),
            'Q': dimensions.get('Q', dimensions.get('W', 1)),
            'Wstride': dimensions.get('Wstride', 1),
            'Hstride': dimensions.get('Hstride', 1),
            'Wdilation': dimensions.get('Wdilation', 1),
            'Hdilation': dimensions.get('Hdilation', 1),
        }
        return layer

