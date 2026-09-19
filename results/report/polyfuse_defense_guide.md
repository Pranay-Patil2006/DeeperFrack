# PolyFuse Interview Defense Guide

> [!IMPORTANT]
> This guide is organized around your **exact resume bullets**. Every section defends a specific phrase. Every claim is verified against the actual source code and output logs.

---

## 📌 Your Resume Bullets (Reference)

> **①** Architected **PolyFuse**, an analytical optimizer scheduling fused-CNN layers on spatial accelerators via a **3-stage pipeline** (tiling, routing, partitioning), replacing brute-force DSE with globally optimal mappings over all contiguous layer sub-stacks.
>
> **②** Devised an **O(N²) DP partitioner** over all contiguous layer sub-stacks with a composite cost coupling **LoopTree energy** with an **SLSQP-based NoC penalty** and **disjoint working-set capacity constraints** across a **4-level memory hierarchy** to eliminate DRAM spills.
>
> **③** Benchmarked on AlexNet/Simba via Timeloop-Accelergy; cut mapping time **16–20×** vs. DeepFrack (5–7 min vs. ~6 hrs), reduced **DRAM traffic** by keeping fused intermediates in on-chip SRAM, and achieved **1.54×** lower energy than the naive baseline.

---

## 🚀 Elevator Pitch (30 Seconds)

> "I built PolyFuse — an analytical optimizer for mapping fused CNN layers onto spatial hardware accelerators. The core problem is the Memory Wall: off-chip DRAM costs ~100× more energy per access than on-chip SRAM. The solution is layer fusion — pipe multiple layers together so intermediate feature maps stay on-chip. The hard part is finding *which* layers to fuse and *how big* to tile them. Prior tools like DeepFrack ran full hardware simulations for every candidate configuration — ~6 hours. PolyFuse replaces that with a 3-stage analytical pipeline that runs in 5–7 minutes, achieving 1.54× better energy than naive execution."

---

## 🖍️ Bullet ① Defense

### "analytical optimizer… replacing brute-force DSE"

**Q: What was DeepFrack doing for 6 hours?**

DeepFrack invokes **Timeloop-Mapper** as an external subprocess for each candidate tile configuration. Timeloop-Mapper runs an internal stochastic search (simulated annealing over loop-nest orderings) to find the best mapping. Each invocation takes seconds to minutes. Doing this across every combination of tile sizes and layer groupings compounds to ~6 hours.

**Q: What does PolyFuse do instead?**

For tile sizing, PolyFuse uses a **closed-form polyhedral recurrence** — the relationship between input and output tile sizes in a convolution is exact geometry:
$$T_{in} = (T_{out} - 1) \times S + K$$
Propagate this backwards through all fused layers and you know every required tile size exactly. Then **binary-search** on $T_{out}$ to find the maximum feasible value under buffer constraints in $O(\log P)$ steps — no simulation at all.

---

### "3-stage pipeline (tiling, routing, partitioning)"

| Stage | File | What it does |
|---|---|---|
| **Stage 1: Tiling** | `stage1_tiler.py` | Binary search to find max tile size; greedy channel tiling; weight cache pinning |
| **Stage 2: Routing** | `stage2_router.py` | SLSQP minimizes squared PE distances; produces NoC proxy penalty for DP |
| **Stage 3: Partitioning** | `stage3_dp.py` | O(N²) DP finds globally optimal fusion grouping |

---

### "globally optimal mappings over all contiguous layer sub-stacks"

**Q: What does globally optimal mean here?**

For $N$ layers, there are $2^{N-1}$ ways to partition them into contiguous fused groups. The DP evaluates every contiguous sub-stack `layers[j:i]` and guarantees the minimum total cost across all valid partitions — not a greedy or heuristic approximation. For AlexNet ($N=5$), this is 16 possible partitions.

**Q: Why only contiguous?**

Fusion requires data to flow from one layer directly into the next. Non-contiguous fusion would require re-fetching data from DRAM, defeating the purpose. Contiguity is a physical constraint of the pipeline, not a simplification.

---

## 🖍️ Bullet ② Defense

### "O(N²) DP partitioner over all contiguous layer sub-stacks"

**The recurrence — write this on the whiteboard:**

$$dp[0] = 0$$
$$dp[i] = \min_{0 \le j < i} \left( dp[j] + \text{Cost}(j,\ i) \right)$$

- `dp[i]` = minimum total energy to optimally process layers $[0..i-1]$
- Backtrack through a `parent[]` array to reconstruct the chosen partition
- $O(N^2)$ subproblems; each calls Stage 1 + LoopTree, so total wall time is dominated by subprocess calls (~10–30 sec each)

---

### "composite cost coupling LoopTree energy with an SLSQP-based NoC penalty"

**Q: Why do you need two cost terms? Why not just use LoopTree energy?**

LoopTree models energy from the **memory hierarchy only** — reads and writes at each buffer level, using the Accelergy-generated ERT. The Simba architecture YAML (`simba_like.yaml`) defines no NoC component, so Accelergy generates **zero ERT entries for inter-PE communication energy**. That cost is completely invisible to LoopTree.

The SLSQP routing stage fills that gap. It assigns each layer a physical $(X_i, Y_i)$ location on the PE mesh and minimizes:
$$\min_{X, Y} \sum_{i=0}^{N-2} \left[(X_{i+1}-X_i)^2 + (Y_{i+1}-Y_i)^2\right]$$

The resulting squared-distance cost is a **proxy for the missing NoC energy**, penalizing partitions where consecutive fused layers would execute on distant PEs. It is added to the LoopTree energy in the DP:

$$\text{Cost}(j, i) = E_{\text{LoopTree}} + 5.0 \times \sum_k \left(\Delta X_k^2 + \Delta Y_k^2\right)$$

**Q: Why squared distance and not linear?**

Squaring makes large hops disproportionately expensive (a 4-hop gap costs 16×, not 4×), naturally clustering consecutive layers tightly on the PE mesh.

---

### "disjoint working-set capacity constraints across a 4-level memory hierarchy"

**Q: What is a working set here?**

The working set of a fused stack is the total set of live tensors that must simultaneously reside in the memory hierarchy during execution: the input to the first layer, all intermediate feature maps, and the output of the last layer.

**Q: What are the 4 levels?**

From the actual Simba arch YAML (`simba_like.yaml`):

| Level | Component | Capacity (AlexNet run) |
|---|---|---|
| L4 — off-chip | DRAM | Unlimited (DDR4) |
| L3 — on-chip shared | GlobalBuffer (`smartbuffer_SRAM`) | 128 KB / 65,536 words |
| L2 — per-PE local | PEInputBuffer / PEWeightBuffer / PEAccuBuffer | 8192 / 32768 / 1024 words |
| L1 — register file | PEWeightRegs | 64 words |

**Q: What does "disjoint" mean?**

Each constraint is enforced independently — not merged into a single inequality. Stage 1 checks three separate constraints per candidate tile:

**Constraint 1 — PEAccuBuffer:** at least one output channel tile must fit in the PE's accumulator:
$$\left\lfloor \frac{\text{cap\_pe\_output}}{T_{out}^2} \right\rfloor \ge 1$$

**Constraint 2 — PEInputBuffer:** at least one input channel tile must fit in the PE's input buffer:
$$\left\lfloor \frac{\text{cap\_pe\_input}}{T_{in}^2} \right\rfloor \ge 1$$

**Constraint 3 — GlobalBuffer:** all simultaneously live feature maps must fit in the shared L3:
$$\underbrace{\sum_{\text{intermediate}} T_{out}^{(i)2} \cdot M_i}_{\text{inter-layer fmaps}} + \underbrace{T_{in,first}^2 \cdot C_{first}}_{\text{first input tile}} + \underbrace{T_{out,last}^2 \cdot M_{last}}_{\text{final output tile}} \le \text{cap\_global}$$

If any single constraint fails → the fusion is hard-rejected. PolyFuse never silently mutates tile sizes to force a fit (a correctness bug in prior tools).

---

### "to eliminate DRAM spills"

**Q: What is a DRAM spill?**

If the working set of a fused stack exceeds GlobalBuffer capacity, intermediate feature maps must be evicted to DRAM mid-computation — reintroducing the expensive DRAM read/write that fusion was designed to avoid. This is exactly analogous to a cache thrash in a CPU's LLC.

**Q: How does PolyFuse eliminate it?**

Constraint 3 is the spill guard. If the total live footprint exceeds `cap_global`, Stage 1 returns `False` and the DP splits the stack shorter. Since a single-layer stack always fits, the DP is guaranteed to find a spill-free partition for any network.

---

## 🖍️ Bullet ③ Defense

### "cut mapping time 16–20× vs. DeepFrack (5–7 min vs. ~6 hrs)"

**Q: Where do these numbers come from?**

- **6 hours:** DeepFrack's reported benchmarking time from their published paper
- **5–7 min:** Measured from PolyFuse's own runs on AlexNet/Simba
- **16–20×:** Published in our IEEE paper comparing the two

**Q: Why does PolyFuse still take 5–7 minutes if it's analytical?**

Stage 1 (binary search) and Stage 2 (SLSQP) are microseconds to milliseconds each. The runtime is dominated by Stage 3's LoopTree subprocess calls (~10–30 sec per evaluated block). But PolyFuse only evaluates DP-selected candidate stacks, not the full exponential DSE space — for AlexNet that is 15 evaluations vs. thousands.

**Q: Is your DeepFrack comparison against the actual tool?**

No — the energy comparison simulates DeepFrack's greedy strategy (try fusing all layers into one block). The timing comparison uses DeepFrack's paper-reported time. We did not run the actual DeepFrack binary.

---

### "reduced DRAM traffic by keeping fused intermediates in on-chip SRAM"

**Q: Exactly how does fusion reduce DRAM traffic?**

In naive execution, every layer writes its output to DRAM and the next layer reads it back. For a 4-layer fused block (AlexNet run5: layers 0–3), that eliminates 3 DRAM write + 3 DRAM read round-trips for the intermediate feature maps (`Inter_0`, `Inter_1`, `Inter_2`). These intermediates instead stay in GlobalBuffer. At ~6 pJ/access (SRAM) vs. ~512 pJ/access (DRAM), that is ~85× cheaper per access — the primary source of the 1.54× energy gain.

---

### "1.54× lower energy than the naive baseline"

**Q: Where does 1.54× come from?**

From the IEEE paper: Naive = 5.76 mJ (Timeloop-Mapper per layer), PolyFuse = 3.75 mJ (LoopTree on DP-optimal partition). $5.76 / 3.75 = 1.536 \approx 1.54\times$.

**Q: Why does the naive baseline use Timeloop-Mapper while PolyFuse uses LoopTree?**

- **Naive:** Each layer evaluated individually by Timeloop-Mapper (full stochastic search for best single-layer tiling) — gives the strongest possible baseline. Results are MD5-cached.
- **PolyFuse:** Tile sizes are analytically determined; LoopTree just evaluates the given mapping (no search, pure cost accounting).

---

## ⚡ Rapid-Fire Q&A

**Q: Why binary search and not a direct algebraic solve?**
A: The feasibility check propagates the recurrence through all $N$ fused layers and checks three separate buffer constraints simultaneously. No closed-form inverse exists for this multi-layer system. But it is monotone in $T_{out}$, so binary search finds the maximum in $O(\log P)$ checks.

**Q: What does Accelergy produce?**
A: An Energy Reference Table (ERT) — a YAML mapping each hardware component to pJ costs per read, write, and update, looked up from technology-node libraries (e.g., `smartbuffer_SRAM` at 45nm). LoopTree multiplies access counts by ERT values to get total energy in pJ.

**Q: What bugs from prior tools did you fix?**
A: Two critical ones:
1. **Overwrite bug:** Prior tools used `=` instead of `+=` for weight footprint accumulation, making 3.2 MB appear to fit in a 512 KB buffer.
2. **Silent tile mutation:** Prior tools shrank tile sizes in a `while` loop to force a fit, silently corrupting the convolution's receptive field. PolyFuse returns a hard `False` — never mutating tile dimensions.

**Q: Would the NoC penalty matter more on a different architecture?**
A: Yes. On larger chiplet designs (64+ chiplets with a defined inter-chiplet network), inter-PE communication is a significant energy fraction and the SLSQP penalty becomes the right mechanism to model it — assuming the arch YAML defines a network component so the scaling constant can be calibrated against a real ERT entry instead of the empirical 5.0 used here.
