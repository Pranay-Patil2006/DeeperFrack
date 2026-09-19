# PolyFuse V2

Analytical CNN layer fusion optimizer for spatial DNN accelerators. Built as part of an SRIP internship project at IIT Gandhinagar.

The core idea: instead of running a massive brute-force design-space search to find how to tile and schedule fused CNN layers onto an accelerator, use algebra to narrow the feasible tile sizes down to a single binary-search pass, then verify with LoopTree. This gets the same quality mapping as DeepFrack in 5–7 minutes instead of ~6 hours.

**Tested on:** AlexNet + Simba-like accelerator (Timeloop/Accelergy ecosystem).

---

## Results (AlexNet on Simba)

| Method | Energy | vs Naive |
|---|---|---|
| Naive (layer-by-layer) | 5.76 mJ | baseline |
| DeepFrack | 3.88 mJ | 1.49x |
| **PolyFuse V2** | **3.75 mJ** | **1.54x** |

Search time: **5–7 min** vs DeepFrack's ~6 hours → **30–40x faster**.

Energy is comparable to DeepFrack while the search time is massively reduced. The key win is eliminating the brute-force benchmarking phase entirely.

---

## How it works

Layer fusion keeps intermediate feature maps on-chip between consecutive CNN layers ("pyramid fusion"), avoiding expensive DRAM round-trips for intermediate activations. The tricky part is deciding *which* layers to fuse together and *what tile size* to use — the space of possibilities is enormous.

PolyFuse V2 cuts through this in three stages:

### Stage 1 — Analytical Tiling
Given a proposed stack of layers, compute the maximum output tile size that fits in on-chip memory without overflowing any buffer. No search over tile sizes; just binary search + the polyhedral recurrence:

```
T_in[i] = (T_out[i] - 1) * stride[i] + R[i]
```

Three buffer constraints are checked simultaneously: PE output buffer, PE input buffer, and the GlobalBuffer (sum of all live feature map tiles across the fused stack). Buffer capacities come from parsing the architecture YAML — nothing is hardcoded.

A greedy knapsack decides which layers' weights to cache on-chip.

### Stage 2 — Spatial Routing
Maps each layer in the fused stack to a physical `(X, Y)` coordinate on the PE mesh, minimizing total squared NoC hop distance. Solved via SLSQP (`scipy.optimize`). Falls back to linear assignment if the solver diverges.

> **Note:** The causality constraint in Stage 2 is currently a placeholder. Implementing proper Extreme Vector / Farkas-based causality bounds is on the TODO list.

### Stage 3 — DP Partitioner
O(N²) dynamic programming over the full layer chain. For each candidate block `[j, i)`:

1. Stage 1 fast-rejects buffer overflow → cost = ∞, skip immediately
2. LoopTree (pytimeloop) evaluates the block's energy and cycles
3. Stage 2 routing penalty is added
4. DP picks the minimum-cost assignment

Backtracking through `parent[]` reconstructs the optimal partition.

The CLI runs all three baselines (Naive via Timeloop mapper, DeepFrack via greedy fuse-all, PolyFuse via DP) on the **exact same architecture YAML** to keep the comparison scientifically honest.

---

## Approaches I tried that didn't work

The problem of scheduling nested for-loops is well-studied in compiler theory. Works like PLUTO model an N-deep loop nest as an N-dimensional polyhedron and transform it to minimize a cost function. This works well for CPUs and GPUs.

I tried to apply the same polyhedral approach to find an optimal mapping completely analytically — no LoopTree calls at all. The math works out cleanly for 2–3 layers but the number of variables explodes beyond 3–4 layers, and the integer programming becomes intractable. Switched to the current hybrid (analytical tiling + LoopTree verification) which is less elegant but actually runs.

I also prototyped a version that used Farkas' Lemma to generate causality constraints for the spatial router. The number of auxiliary variables grew to 50,000+ for a realistic convolution kernel, making it completely unusable. The "Extreme Vector" method (checking only the corner vectors of the feasibility polytope) is the right approach here, but I haven't finished wiring it in yet.

Earlier versions of the tiler had a bug inherited from the DeepFrack codebase: weight buffer sizes were being overwritten (`=`) rather than accumulated (`+=`) across layers in the stack. This made it look like you could cache 3.2 MB of weights in a 512 KB buffer. Fixed.

---

## Repo layout

```
.
├── polyfuse/
│   ├── cli.py                   # entry point: python3 -m polyfuse.cli
│   ├── hal.py                   # parses arch + components YAMLs → HardwareConfig
│   ├── stage1_tiler.py          # binary search + polyhedral recurrence
│   ├── stage2_router.py         # SLSQP spatial PE assignment
│   ├── stage3_dp.py             # O(N²) DP partitioner
│   ├── looptree_runner.py       # fused workload + mapping YAML gen, LoopTree subprocess
│   ├── timeloop_integration.py  # Timeloop-mapper calls for naive baseline
│   ├── reporter.py              # result formatting
│   └── cpp_parsers/             # pybind11 wrapper for Timeloop's C++ workload parser
├── data/
│   ├── alexnet/                 # 4 AlexNet layer YAMLs (Timeloop problem format)
│   └── components/              # Accelergy component YAMLs (Simba target)
├── docs/
│   ├── master_plan.md           # full spec: HAL, all three stages, regression tests
│   ├── PolyFuseV2_Explanation.md
│   └── polyfuse_v2_presentation.pdf
├── tests/                       # unit + regression tests
├── results/                     # saved benchmark outputs and LoopTree logs
├── legacy/                      # PolyFuse V1, PLUTO experiments, earlier prototypes
├── comprehensive_benchmark.py   # batch run: AlexNet, VGG02, MobileNet vs all baselines
├── run_actual_deepfrack.py      # emulates the DeepFrack partition, evaluates via LoopTree
├── run_compare.py               # head-to-head comparison helper
├── generate_final_report.py     # post-hoc markdown report from saved results
└── setup.py                     # pybind11 extension build (timeloop_pybind.so)
```

The `legacy/` folder has PolyFuse V1 (pure analytical, no LoopTree verification), a PLUTO-style polyhedral prototype, and a bunch of intermediate test scripts. Left in because the iteration history is useful context.

---

## Running it

**Prerequisites**: Timeloop + Accelergy installed (tested with the `accelergy-timeloop-infrastructure` Docker image), `pytimeloop`, `scipy`, `numpy`, `pyyaml`.

```bash
python3 -m polyfuse.cli \
  --arch       /path/to/arch/simba_like.yaml \
  --components data/components/ \
  --network    data/alexnet/ \
  --output     results/run_01/ \
  --objective  energy
```

Objectives: `energy` (default), `latency`, `edp`.

The naive baseline is cached to disk after the first run (`naive_cache.json`) — Timeloop mapper calls are slow, no point repeating them.

```bash
# Re-run just the DeepFrack emulation
python3 run_actual_deepfrack.py

# Batch benchmark (AlexNet, VGG02, MobileNet)
python3 comprehensive_benchmark.py
```

**Expected output:**
```
=================================================================
  PolyFuse V2 — End-to-End Energy Comparison
=================================================================
  Method                               Energy (pJ)    vs Naive
  Naive (layer-by-layer, tiled)          5.76e+12     baseline
  DeepFrack (greedy fuse-all)            3.88e+12       1.49x
  PolyFuse (optimal DP)                  3.75e+12       1.54x
=================================================================
  PolyFuse Optimal Partition:
    Block 0: layer01 -> layer02 -> layer03
    Block 1: layer04
```

---

## TODO

### Near-term
- [ ] Wire in proper Extreme Vector causality constraints in `stage2_router.py` — the current placeholder always returns feasible which means the routing penalty is slightly incorrect
- [ ] Fix the hardcoded `naive_cache.json` path in `timeloop_integration.py` (currently points to the old `PolyFuseV2/` path)
- [ ] Add `--mapper` flag so the entry point can target the Timeloop mapper for PolyFuse blocks too (not just LoopTree)
- [ ] Write actual unit tests in `tests/` — currently empty

### Medium-term
- [ ] Extend to VGG02 and MobileNetV2 (benchmark scripts exist but rely on paths from the Docker image)
- [ ] Eyeriss and Gemmini architecture support (HAL should handle them already via YAML parsing, untested)
- [ ] Proper latency model in Stage 3 — currently proxied as `base_cycles` from LoopTree, which doesn't account for pipeline stalls at stack boundaries
- [ ] Halo recompute vs. store decision: currently always stores. Should analytically decide based on `Vol_halo × Cost_MAC` vs `Vol_spill × Cost_DRAM`
- [ ] ResNet / skip-connection support — the DP assumes a linear chain, residual blocks would need a DAG extension

### Longer-term / research directions
- [ ] Revisit the polyhedral approach for shallow networks (≤ 3 layers) where the variable count stays manageable
- [ ] Newton-Raphson solver for non-quadratic buffer constraints (needed for skip connections where the footprint equation degree > 2)
- [ ] PE array quantization: round tile sizes down to PE array multiples, reallocate the SRAM slack to channel tiles

---

## References

1. Parashar, A. et al. "Timeloop: A Systematic Approach to DNN Accelerator Evaluation." *ISPASS 2019.* https://ieeexplore.ieee.org/document/8695666

2. Wu, Y. N. et al. "Accelergy: An Architecture-Level Energy Estimation Methodology for Accelerator Designs." *ICCAD 2019.* https://ieeexplore.ieee.org/document/8942149

3. Gilbert, M. et al. "LoopTree: Enabling Exploration of Fused-Layer Dataflow Accelerators." *ISPASS 2023.* https://ieeexplore.ieee.org/document/10158185

4. Glint, T., Pechimuthu, M., and Mekie, J. "DeepFrack: A Comprehensive Framework for Layer Fusion, Face Tiling, and Efficient Mapping in DNN Hardware Accelerators." *DATE 2024.*

5. Bondhugula, U. et al. "A Practical Automatic Polyhedral Parallelizer and Locality Optimizer." *PLDI 2008.* https://dl.acm.org/doi/10.1145/1375581.1375595

6. Shao, Y. S. et al. "Simba: Scaling Deep-Learning Inference with Multi-Chip-Module-Based Architecture." *MICRO 2019.* https://dl.acm.org/doi/10.1145/3352460.3358302
