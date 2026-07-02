import os
import tempfile
import sys
from polyfuse.looptree_runner import evaluate_single_block


class Stage3DP:
    def __init__(self, tiler, router, hw_config, objective='energy'):
        self.tiler     = tiler
        self.router    = router
        self.hw_config = hw_config   # Fix 2&3: carry real hw_config through
        self.objective = objective

    def partition(self, layers):
        n = len(layers)

        dp        = [float('inf')] * (n + 1)
        dp[0]     = 0
        parent    = [-1] * (n + 1)
        best_meta = [None] * (n + 1)

        for i in range(1, n + 1):
            for j in range(i):
                stack         = layers[j:i]
                layer_indices = list(range(j, i))

                # --- Stage 1: fast capacity check (no LoopTree invocation) ---
                is_valid, tile_sizes, weight_pattern, _ = \
                    self.tiler.evaluate_stack(stack)

                if not is_valid:
                    continue

                # --- Stage 3: accurate LoopTree evaluation ---
                with tempfile.TemporaryDirectory() as tmp_dir:
                    print(f"[PolyFuse] Evaluating block {layer_indices} with LoopTree...")
                    sys.stdout.flush()

                    stack_layers = [layers[idx] for idx in layer_indices]
                    # Pass hw_config so the worker uses the REAL Simba arch
                    base_energy = evaluate_single_block(
                        stack_layers, layer_indices, tile_sizes,
                        tmp_dir, hw_config=self.hw_config)

                    print(f"[PolyFuse] Block {layer_indices} energy: {base_energy:.4e} pJ")

                if base_energy == float('inf'):
                    continue

                # Stage 2: spatial routing cost
                routing      = self.router.optimize_spatial_routing(stack)
                routing_cost = 0.0
                for r_idx in range(len(routing) - 1):
                    x1, y1 = routing[r_idx]
                    x2, y2 = routing[r_idx + 1]
                    routing_cost += ((x2 - x1)**2 + (y2 - y1)**2) * 5.0

                total_cost = base_energy + routing_cost

                if self.objective == 'latency':
                    total_cost = total_cost / 100.0
                elif self.objective == 'edp':
                    latency    = total_cost / 100.0
                    total_cost = total_cost * latency

                if dp[j] + total_cost < dp[i]:
                    dp[i]     = dp[j] + total_cost
                    parent[i] = j
                    best_meta[i] = {
                        'stack':          [layer['name'] for layer in stack],
                        'tile_sizes':     tile_sizes,
                        'weight_pattern': weight_pattern,
                        'routing':        routing,
                        'cost':           total_cost,
                    }

        # --- Reconstruct optimal partition ---
        if dp[n] == float('inf'):
            return None  # no valid partition exists

        partition = []
        curr = n
        while curr > 0:
            p = parent[curr]
            partition.insert(0, best_meta[curr])
            curr = p

        return partition
