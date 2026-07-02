# PolyFuse V2: Master Plan
*A standalone, end-to-end analytical CNN layer fusion optimizer for spatial accelerators.*

---

Use scientific thinking and critical judgement while building this.

## 0. System Interface & CLI

PolyFuse V2 is invoked as a single command-line tool that takes Timeloop-format inputs and produces optimized fused-layer schedules with verified energy results.

**Usage:**
```bash
polyfuse \
  --arch /path/to/arch.yaml \
  --components /path/to/components/ \
  --network /path/to/layers/ \
  --output /path/to/output/ \
  --objective energy            # energy | latency | edp
  --verbose
```

**Inputs (all Timeloop format):**
*   `--arch`: The Timeloop architecture YAML (e.g., `simba_like.yaml`). Defines the full memory hierarchy, PE array dimensions, buffer depths/widths, and NoC topology.
*   `--components`: The Accelergy component directory (e.g., `smartbuffer_SRAM.yaml`, `lmac.yaml`). Defines energy-per-action tables for every hardware primitive.
*   `--network`: A directory of Timeloop problem YAMLs, one per layer (e.g., `layer01.yaml`, `layer02.yaml`, ...), defining the convolution dimensions ($K, C, H, W, R, S$, stride, dilation). Layers are assumed to form a **linear chain** in filename order.

**Outputs (written to `--output`):**
*   `schedule.yaml` — The optimal fused-layer mapping in Timeloop-compatible format.
*   `report.md` — Human-readable report with partition table, per-layer energy breakdown, and comparison vs. naive baseline.
*   `results.json` — Machine-readable results for downstream tooling.
*   `naive_baseline/` — Timeloop stats for each layer run independently (SLC mode).
*   `fused_stacks/` — Timeloop stats for each fused stack in the optimal partition.

---

## 1. Data Ingestion: Hardware Abstraction Layer (HAL)

All hardware parameters must be **parsed from the Timeloop architecture and component files**, never hardcoded.

### 1.1 Architecture Parser
Read `arch.yaml` and extract:
*   **Buffer inventory:** For each storage level, extract `name`, `memory_depth`, `memory_width`, `word-bits`, instance count (from `[0..N]` notation), and `cluster-size`.
*   **Buffer capacity (in words):** Computed as `depth × width / word-bits` per instance, then aggregated across bank instances.
*   **Buffer role assignment:** Determine which buffers hold inputs, weights, and outputs based on the Timeloop constraint files and the architecture's dataflow mapping.
*   **PE array geometry:** Extract `meshX` and `meshY` to determine the spatial PE grid dimensions.
*   **Memory hierarchy depth:** Enumerate levels from registers → scratchpad → global buffer → DRAM.

### 1.2 Component Energy Extraction
Read `components/*.yaml` and extract:
*   Energy-per-access for each buffer level (read/write/leak).
*   Energy-per-MAC from the compute primitive (e.g., `lmac.yaml`).
*   These values populate the analytical energy proxy used for rapid structural optimization in Stage 1.

### 1.3 Network Loader
Read each `layer_XX.yaml` from the network directory and extract:
*   $K$ (output channels), $C$ (input channels)
*   $H, W$ (output spatial dimensions)
*   $R, S$ (kernel height/width)
*   $U, V$ (stride height/width)
*   $Dilation$ (if present)
*   Layers are ordered sequentially to form a linear fusion chain.

---

## 2. Three-Stage Core Architecture

### Stage 1: Zero-Search Algebraic Tiling

**Mechanism:** Determine the mathematically optimal tile sizes without any iterative search or simulation.

**Algorithm:**
1.  Propagate required input footprints backward from the output tile dimension using the exact polyhedral recurrence:
    $$T_{in}^{(i)} = (T_{out}^{(i)} - 1) \times S^{(i)} + R^{(i)}$$
    Since layers are sequentially fused, $T_{out}^{(i-1)} = T_{in}^{(i)}$.

2.  For complex topologies (e.g., ResNets with skip connections) where the buffer capacity equation degree exceeds 2, replace the static quadratic formula with a fast **Newton-Raphson** numeric solver.

**Multi-Level Memory:**
Solve deeper hierarchies (e.g., L2 Global Buffer → L1 Scratchpad) using decoupled, cascaded quadratic root-finding. Guarantee perfect mathematical nesting:
$$ T_{L1} = \max \{ d \in \mathbb{N} \mid d \le T_{L1_{float}} \land (T_{L2} \bmod d = 0) \} $$

**Buffer Constraint Checking:**
Enforce strict, per-buffer hardware limits as algebraic inequalities. For every intermediate layer $i$ in a fused stack, **all three** constraints must hold simultaneously:
1.  **Input Buffer:** $[T_{in}^{(i)}]^2 \times C^{(i)} \le \text{InputBuffer}_{capacity}$
2.  **Output Buffer:** $[T_{out}^{(i)}]^2 \times M^{(i)} \le \text{AccuBuffer}_{capacity}$
3.  **Weight Buffer (if cached):** $\sum_{j \in Cached} R^{(j)} \times S^{(j)} \times C^{(j)} \times M^{(j)} \le \text{WeightBuffer}_{capacity}$

Buffer capacities are parsed from the HAL (Section 1), never hardcoded.

**PE Array Quantization:**
After computing the maximum algebraic tile size, round down to the nearest PE array multiple to guarantee 100% PE utilization. Reallocate the resulting SRAM slack to increase channel tile dimensions ($T_k$ or $T_c$), or absorb expanding intermediate halo data for deeper fusions.

**Weight Caching Strategy:**
Use a greedy knapsack algorithm to pack as many layers' weights into the weight buffer as possible without overflowing. Weight sizes must be **accumulated** ($+=$), never overwritten ($=$).

### Stage 2: Hardware-Aware Spatial Routing (MIQP)

**Mechanism:** Map valid layer stacks spatially across the NoC processing elements while ensuring data causality.

**Routing Penalty — Convex Quadratic Programming:**
Minimize physical NoC hop distances using a convex squared-distance objective:
$$J = \min_{X, Y} \sum_{i=start}^{end-1} \left( X^{(i+1)} - X^{(i)} \right)^2 + \left( Y^{(i+1)} - Y^{(i)} \right)^2$$
Where $(X^{(i)}, Y^{(i)})$ are the 2D NoC coordinates for the PE executing layer $i$.

This convex formulation:
- Requires **zero** dummy binary variables (unlike absolute-value linearization)
- Naturally penalizes long hops quadratically (distance 4 costs 16×, distance 1 costs 1×)
- Solves in < 2ms via `scipy.optimize.minimize(SLSQP)`

**Causality — Extreme Vector Method (Bypassing Farkas' Lemma):**
For a $3 \times 3$ kernel, enforce causality at the 4 geometric corner vectors:
$$\Theta_2 \cdot \vec{x} - \Theta_1 \cdot (\vec{x} - \vec{d}_{ext}) \ge 1 \quad \forall \vec{d}_{ext} \in \{(0,0), (0,2), (2,0), (2,2)\}$$

This reduces constraints from 50,000+ Farkas multiplier variables down to ~4–8 linear equations per layer boundary.

**Spatial Hardware Pinning:**
Bound the spatial execution to the physical PE grid (parsed from HAL):
$$0 \le \Theta^{SpaceX} \cdot \vec{x} < PE_{width}$$
$$0 \le \Theta^{SpaceY} \cdot \vec{x} < PE_{height}$$

### Stage 3: Dynamic Programming (DP) Partitioning

**Mechanism:** Find the globally optimal partition of the full linear chain into fused stacks.

**Algorithm:** Run an $O(N^2)$ DP pass:
$$DP[i] = \min_{0 \le j < i} \left( DP[j] + \text{EnergyCost}(\text{Stack}(j, i)) \right)$$

Where `EnergyCost(Stack(j, i))` is the total cost of fusing layers $j$ through $i$, computed using the analytically determined tile sizes and weight caching patterns from Stage 1, spatially optimized by Stage 2.

**Recompute vs. Store Decision:**
For each fused stack boundary, analytically decide whether to recompute the halo or store it:
```
if (Vol_halo × Cost_MAC) < (Vol_spill × Cost_DRAM):
    → Recompute (use Index Set Splitting to decouple tiles)
else:
    → Store (share halo data sequentially through SRAM)
```

**Optimization Objective:**
The DP partitioner supports three analytical objective modes (selected via `--objective`):
*   **`energy`** — Minimize total energy: $E_{total} = E_{MAC} + E_{SRAM} + E_{DRAM}$
*   **`latency`** — Minimize total execution cycles, accounting for compute-bound vs. memory-bound bottlenecks per tile using analytical roofline modeling.
*   **`edp`** — Minimize the Energy-Delay Product: $EDP = E_{total} \times Latency$

All three modes use purely analytical mathematical models — no genetic algorithms, machine learning, or stochastic methods.

---

## 3. Timeloop Integration Layer

PolyFuse V2 invokes Timeloop programmatically for final verification. This is **not** a pre-computed lookup — it dynamically generates and runs Timeloop for each candidate.

### 3.1 Dynamic YAML Generation
For each fused stack in the optimal partition, the Timeloop Integration Layer:
1.  Generates a virtual `prob.yaml` with tile-adjusted dimensions (including halo expansion for redundant compute).
2.  Selects the correct constraint file based on the layer's position in the stack and caching pattern:
    *   **Start layer:** `Start.yaml` (or `OutWCC.yaml` if weights cached)
    *   **Middle layers:** `LBLC.yaml` (or `WCC.yaml` if weights cached)
    *   **End layer:** `ELBLC.yaml` (or `EWCC.yaml` if weights cached)
    *   **Single layer:** `SLC.yaml`
3.  Assembles the full Timeloop command with the architecture, components, problem, constraints, and mapper config.

### 3.2 Timeloop Invocation
Invoke `timeloop-mapper` as a subprocess for each generated configuration. Parse the output `stats.txt` to extract:
*   Total energy (uJ)
*   Per-level energy breakdown (DRAM, GlobalBuffer, PE buffers, MACs)
*   Cycle count / latency

### 3.3 Automated Naive Baseline
Automatically generate and run Timeloop for every layer independently using `SLC` (Single Layer Computation) constraints. This produces the **naive baseline** — the energy of executing each layer in isolation with full DRAM round-trips for intermediate data.

---

## 4. Verification & Reporting

### 4.1 Timeloop-Verified Comparison
The final verification must be done entirely in Timeloop, with a strict layer-by-layer comparison:

| Metric | Naive (SLC) | PolyFuse V2 (Fused) | Reduction |
|:---|---:|---:|---:|
| Layer 1 Energy | $E_{naive,1}$ | — | — |
| Layer 2 Energy | $E_{naive,2}$ | — | — |
| ... | ... | — | — |
| **Stack (0, K) Fused** | — | $E_{fused,stack1}$ | — |
| **Stack (K+1, N) Fused** | — | $E_{fused,stack2}$ | — |
| **Total** | $\sum E_{naive}$ | $\sum E_{fused}$ | **X.XX%** |

### 4.2 Output Artifacts
The optimizer produces:
*   **`report.md`**: Human-readable summary with the partition table, per-stack tile sizes, weight caching patterns, energy breakdown, and reduction percentage vs. naive.
*   **`results.json`**: Machine-readable results containing all numerical data for downstream analysis.
*   **`schedule.yaml`**: The optimal fused-layer mapping exported in Timeloop-compatible format.
*   **Per-stack Timeloop logs**: Full `stats.txt` output for each fused stack and each naive layer, stored in subdirectories for reproducibility.

### 4.3 Correctness Invariants
The optimizer must enforce:
*   No buffer overflow: every tile/caching combination must satisfy all three buffer constraints.
*   No receptive field truncation: intermediate tile sizes must never be shrunk to fit — the stack must be rejected if the footprint exceeds capacity.
*   Weight accumulation: cumulative weight sizes must use `+=`, never `=`.
*   Energy reduction must be non-negative: if fusion costs more than naive, the DP must select the naive (unfused) partition.

---

## 5. Benchmark Suite & Regression Testing

### 5.1 Network Benchmark Suite
| Network | Layers | Notes |
|:---|:---:|:---|
| AlexNet | 4 | Primary regression target (existing YAMLs) |
| VGG02 | 12 | Deep chain, heavy weight pressure |
| MobileNetV2 | ~53 | Depthwise separable convolutions |
| ResNet-50 | ~50 | Skip connections (linear chain segments between residual blocks) |
| EfficientNet-B0 | ~82 | MBConv blocks with squeeze-and-excitation |

### 5.2 Architecture Targets
| Architecture | Type | Notes |
|:---|:---|:---|
| Simba | Chiplet NoC, Weight-Stationary | Primary target (existing YAMLs) |
| Eyeriss | Spatial array, Row-Stationary | Different dataflow, different buffer structure |
| Gemmini | Systolic array, WS/OS configurable | Open-source, supports dataflow selection |

### 5.3 Automated Comparison
For each (network, architecture) pair, automatically produce:
*   Naive baseline energy (Timeloop SLC)
*   PolyFuse V2 optimized energy (Timeloop verified)
*   DeepFrack energy (where available, from logs)
*   Reduction percentages and optimization wall-clock time

### 5.4 Regression Tests
*   **Buffer overflow test:** Verify that no generated schedule exceeds any buffer capacity.
*   **Weight accumulation test:** Verify cumulative weight sizes match manual calculation.
*   **Receptive field test:** Verify that no intermediate tile is truncated.
*   **Energy monotonicity test:** Verify that the fused schedule never costs more than naive.
*   **Determinism test:** Verify that the optimizer produces identical results on repeated runs.

---

## 6. Known Pitfalls & Resolutions

*   **DeepFrack Overwrite Bug:** Always accumulate weight sizes with `+=`. The `=` overwrite bug allowed DeepFrack to hallucinate caching 3.2MB of weights in a 512KB buffer.
*   **Receptive Field Truncation:** Never shrink an intermediate tile to fit in SRAM. If the back-propagated $T_{in}$ footprint exceeds buffer capacity, reject the stack entirely. DeepFrack's `while Tile_Width -= 1` loop silently broke convolution physics.
*   **Farkas Multiplier Explosion:** Never use standard Farkas' Lemma ILPs for deep spatial convolutions. The Extreme Vector method reduces 50,000+ variables to ~8 constraints.
*   **Padding Non-affinities:** To maintain affine loop bounds, separate boundary tiles from the steady-state center (the Halide/Tiramisu approach), or use DRAM pre-padding.
*   **Buffer Collisions:** Treat spatial hardware buffers as physically disjoint (Input, Weight, Output). Each buffer has its own capacity constraint — never combine them into a single monolithic SRAM inequality.
*   **Energy Model Divergence:** The analytical proxy (used for rapid structural decisions) will diverge from Timeloop's exact numbers. Always use Timeloop as the final ground truth. If the proxy ranks candidates differently than Timeloop, trust Timeloop.

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Build the pipeline end-to-end)
1.  Implement the HAL (Section 1) — parse `arch.yaml` and `components/` into a normalized hardware descriptor.
2.  Implement the Network Loader — parse layer YAMLs into a sequential layer list.
3.  Port and merge Stage 1 (analytical tiling) from the existing `polyfuse_pure.py`, incorporating the output buffer fix from `test_fix.py`.
4.  Implement the Timeloop Integration Layer (Section 3) — dynamic YAML generation, subprocess invocation, result parsing.
5.  Implement the DP partitioner (Stage 3) using HAL-derived energy proxies.
6.  Implement the CLI entry point (Section 0) and the reporting pipeline (Section 4).
7.  Validate end-to-end on AlexNet + Simba.

### Phase 2: Spatial Optimization (Integrate Stage 2)
1.  Port the MIQP spatial routing prototypes (`miqp_scheduler.py`, `miqp_farkas_solver.py`) into the main pipeline.
2.  Wire Stage 2 output into Stage 3's cost function.
3.  Add latency and EDP objective modes to the DP partitioner.
4.  Validate on VGG02 + Simba.

### Phase 3: Generalization (Multiple networks & architectures)
1.  Add PE array quantization and channel tiling to Stage 1.
2.  Extend the benchmark suite to MobileNetV2, ResNet-50, EfficientNet.
3.  Add Eyeriss and Gemmini architecture support.
4.  Implement the full regression test suite (Section 5.4).
5.  Performance comparison against DeepFrack and published results.
