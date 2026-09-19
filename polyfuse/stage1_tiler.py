"""
Stage 1: Zero-Search Analytical Tiler for PolyFuse V2.

Computes the maximum tile sizes that satisfy ALL buffer capacity constraints
using the polyhedral recurrence:
    T_in^(i) = (T_out^(i) - 1) * stride^(i) + R^(i)

Uses ONLY buffer capacities parsed from the HAL — no hardcoded defaults.
"""
import math


class Stage1Tiler:
    def __init__(self, hw_config):
        self.hw_config = hw_config
        self._extract_capacities()

    def _extract_capacities(self):
        """
        Extract relevant capacities dynamically from the parsed HAL hardware config.
        Resolves role capacities to support arbitrary hierarchies (e.g. shared buffers).
        """
        buffers = self.hw_config.buffers

        # Find capacities dynamically by role
        self.cap_dram = 0
        self.cap_global = 0
        self.cap_pe_input = 0
        self.cap_pe_weight = 0
        self.cap_pe_output = 0

        # Find name of global buffer for printing info
        self.global_buf_name = 'GlobalBuffer'

        for name, info in buffers.items():
            role = info.get('role', 'unknown')
            cap = info.get('capacity_words', 0)
            if role == 'dram':
                self.cap_dram = max(self.cap_dram, cap)
            elif role == 'global':
                self.cap_global = max(self.cap_global, cap)
                self.global_buf_name = name
            elif role == 'input':
                self.cap_pe_input = max(self.cap_pe_input, cap)
            elif role == 'weight' and 'reg' not in role:
                self.cap_pe_weight = max(self.cap_pe_weight, cap)
            elif role == 'output':
                self.cap_pe_output = max(self.cap_pe_output, cap)

        # Fallbacks for shared hierarchies (e.g. PE buffers share a single SRAM or global)
        if self.cap_pe_input == 0:
            self.cap_pe_input = self.cap_global
        if self.cap_pe_output == 0:
            self.cap_pe_output = self.cap_global
        if self.cap_pe_weight == 0:
            # Look for weight registers
            for name, info in buffers.items():
                if info.get('role') == 'weight_reg':
                    self.cap_pe_weight = max(self.cap_pe_weight, info.get('capacity_words', 0))
            if self.cap_pe_weight == 0:
                self.cap_pe_weight = self.cap_global

        # Validate essential capacities are present
        if self.cap_global == 0:
            raise ValueError(
                f"HAL failed to identify any shared Global Buffer (role=global) in the hierarchy.\n"
                f"Parsed buffers: {list(buffers.keys())}"
            )

        # PE array geometry
        self.pe_count = self.hw_config.pe_array.get('total_pes', 1)
        self.macs_per_pe = self.hw_config.pe_array.get('macs_per_pe', 1)

        print(f"[Stage1Tiler] Generic Hierarchy Initialized:")
        print(f"  - DRAM Capacity: {'Unlimited' if self.cap_dram == 2**30 else str(self.cap_dram) + ' words'}")
        print(f"  - Global Shared Buffer ({self.global_buf_name}) Capacity: {self.cap_global} words "
              f"({self.cap_global * self.hw_config.buffers.get(self.global_buf_name, {}).get('word_bits', 16) // 8 / 1024:.1f} KB)")
        print(f"  - PE-local Input Capacity: {self.cap_pe_input} words")
        print(f"  - PE-local Weight Capacity: {self.cap_pe_weight} words")
        print(f"  - PE-local Output Capacity: {self.cap_pe_output} words")
        print(f"  - PE array: {self.pe_count} PEs x {self.macs_per_pe} MACs/PE")

    def evaluate_stack(self, layers):
        """
        Evaluate if a stack of layers can be fused.

        Returns (is_valid, tile_sizes, weight_caching_pattern, analytical_energy)

        tile_sizes: dict[layer_name -> {T_out, T_in, T_c, T_m}]
        """
        if not layers:
            return False, {}, [], float('inf')

        # Find the maximum output tile size for the LAST layer that
        # satisfies ALL buffer constraints across the full stack.
        max_t_out = self._find_max_t_out(layers)

        if max_t_out <= 0:
            return False, {}, [], float('inf')

        if max_t_out <= 0:
            return False, {}, [], float('inf')

        # Back-propagate tile sizes through the stack
        tile_sizes = {}
        current_t_out = max_t_out

        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            R = layer['R']
            Hstride = layer['Hstride']
            t_in = (current_t_out - 1) * Hstride + R

            # Dynamic resolution of local tiling constraints (T_c, T_m)
            # PE-local output capacity constraint
            max_t_m = max(1, self.cap_pe_output // (current_t_out * current_t_out))
            t_m = min(layer['M'], max_t_m)

            # PE-local input capacity constraint
            max_t_c = max(1, self.cap_pe_input // (t_in * t_in))
            
            # PE-local weight capacity constraint
            weight_limited_t_c = max(1, self.cap_pe_weight // (t_m * layer['R'] * layer['S']))
            t_c = min(layer['C'], max_t_c, weight_limited_t_c)

            tile_sizes[layer['name']] = {
                'T_out': current_t_out,
                'T_in': t_in,
                'T_c': t_c,
                'T_m': t_m,
            }
            current_t_out = t_in

        # Greedy knapsack weight caching
        weight_pattern = self._compute_weight_caching(layers)

        return True, tile_sizes, weight_pattern, 0.0

    # ------------------------------------------------------------------
    # Constraint checking
    # ------------------------------------------------------------------

    def _find_max_t_out(self, layers):
        """Binary search for the maximum feasible T_out of the LAST layer."""
        high = min(layer['P'] for layer in layers)
        low = 1
        best = 0

        while low <= high:
            mid = (low + high) // 2
            if self._check_constraints(layers, mid):
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        return best

    def _check_constraints(self, layers, t_out_last):
        """
        Verify that the given T_out for the last layer satisfies ALL buffer
        constraints throughout the fused stack.
        """
        current_t_out = t_out_last
        live_fmaps = []

        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            M = layer['M']
            C = layer['C']
            R = layer['R']
            Hstride = layer['Hstride']
            t_in = (current_t_out - 1) * Hstride + R

            # 1. PE-local output buffer constraint
            if self.cap_pe_output > 0:
                max_t_m = self.cap_pe_output // (current_t_out * current_t_out)
                if max_t_m < 1:
                    return False

            # 2. PE-local input buffer constraint
            if self.cap_pe_input > 0:
                max_t_c = self.cap_pe_input // (t_in * t_in)
                if max_t_c < 1:
                    return False

            if i < len(layers) - 1:
                live_fmaps.append(current_t_out * current_t_out * M)

            current_t_out = t_in

        first_layer = layers[0]
        first_t_in = current_t_out
        live_fmaps.append(first_t_in * first_t_in * first_layer['C'])
        live_fmaps.append(t_out_last * t_out_last * layers[-1]['M'])

        # 3. Global shared buffer capacity constraint
        total_gb = sum(live_fmaps)
        if total_gb > self.cap_global:
            return False

        return True

    # ------------------------------------------------------------------
    # Weight caching
    # ------------------------------------------------------------------

    def _compute_weight_caching(self, layers):
        """
        Greedy knapsack: accumulate layer weight sizes until the weight buffer is full.
        """
        pattern = []
        accumulated = 0
        for layer in layers:
            w_size = layer['R'] * layer['S'] * layer['C'] * layer['M']
            if accumulated + w_size <= self.cap_pe_weight:
                accumulated += w_size
                pattern.append(layer['name'])
        return pattern


