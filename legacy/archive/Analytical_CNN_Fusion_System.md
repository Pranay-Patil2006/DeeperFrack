# Analytical CNN Fusion System: Mathematical Hardware-Aware Scheduling

This document synthesizes the complete theoretical and architectural design for an advanced, strictly analytical CNN fusion optimizer. It builds upon polyhedral optimization theory (akin to Pluto) but is fundamentally re-engineered to target spatial hardware accelerators (like Simba and Eyeriss) by incorporating analytical algebra to eliminate iterative searches and machine learning approximations.

---

## 1. System Vision and Inputs

The goal of the system is to mathematically determine the optimal fused layer schedule (maximizing data locality and balancing recomputation vs. memory spills) without relying on guesswork, heuristics, or brute-force search. 

**Inputs (Timeloop-Compatible Interface):**
*   **`arch.yaml`:** Hardware constraints (SRAM sizes, PE array dimensions, MAC energy, DRAM energy, multiple-buffering configurations).
*   **`prob.yaml`:** The full topological CNN workload manually defined by the user.

**Outputs:**
*   A mathematically guaranteed optimal mapping schedule, verified by Timeloop as the ground-truth calculator.

---

## 2. Core Architecture: The Two-Stage Mathematical Solver

To achieve mathematical optimality without causing standard Integer Linear Programming (ILP) solvers to hang on non-linear equations (like Ehrhart buffer volume polynomials), the system employs a decoupled, two-stage solver.

### Stage 1: Zero-Search Analytical Tiling (Data-Space Driven)
Instead of tiling the loops and checking if the data fits, we tile the tensors and solve for the maximum loop bounds algebraically. 
For two fused convolutions (Layer 1 $\rightarrow$ Layer 2), the required input tile size ($T_{in}$) is a linear function of the output tile size ($T_{out}$): $T_{in} = (T_{out} - 1) \cdot S + K$.

We plug this directly into the SRAM capacity constraint:
$$(C_{out} \cdot T_{out}^2) + C_{in} \cdot ((T_{out} - 1) \cdot S + K)^2 + W_{size} \le SRAM\_Capacity$$

This resolves into a standard quadratic equation: $A \cdot T_{out}^2 + B \cdot T_{out} + C \le SRAM\_Capacity$.
*   **The Action:** The compiler executes the quadratic formula to instantly find the absolute maximum floating-point tile size, and takes the `floor()`. 
*   **The Result:** We calculate the 100% optimal integer tile size in microseconds without a single iterative search loop. These constant bounds are passed to Stage 2.

### Stage 2: Hardware-Constrained ILP Scheduling
With the tile sizes fixed as constants (eliminating non-linear complexity), we invoke an ILP solver (via Polyhedral models like ISL) to find the optimal scheduling matrix ($\Theta$) for the fused loops.

The ILP optimizes the schedule using the following constraints and objectives:
1.  **Rectangular Overlapped Tiling:** We strictly constrain the affine transformations to orthogonal, rectangular bounds. This sacrifices some mathematically perfect data reuse to ensure the resulting loops can be executed by rigid hardware FSMs (like Eyeriss) without complex `min/max` control divergence.
2.  **Spatial Hardware Pinning:** The ILP matrix is split into Space and Time. The spatial dimensions are mathematically bounded by the PE array sizes from `arch.yaml`.
3.  **The Recomputation Objective:** The ILP objective function mathematically balances **Overlapped Tiling (Redundant Compute)** vs. **DRAM Spills**.
    *   *Minimize:* $E_{total} = (Cost_{MAC} \cdot Vol_{overlap}(\Theta)) + (Cost_{DRAM} \cdot Vol_{spill}(\Theta))$
    *   For shallow fusions, the math naturally selects overlapped tiling (recalculating the "halo" region) because MACs are incredibly cheap. For deep fusions, the exponential growth of redundant MACs forces the ILP to automatically break the fusion chain and spill to DRAM.

---

## 3. Verification Workflow (Orchestrating Timeloop)

A critical flaw in using Timeloop for fusion is that Timeloop's C++ core expects to map a single operational layer at a time. It cannot natively evaluate a fused schedule with complex inter-layer lifetimes or overlapped redundant compute.

To bypass this without rewriting Timeloop, our system uses Python to dynamically orchestrate the evaluation:

1.  **Virtual Fused Workload Generation:** If the ILP decides that Layer 2 should redundantly recalculate Layer 1's halo, the Python orchestrator dynamically rewrites Timeloop's `prob.yaml`. It creates a "Virtual Fused Layer" whose MAC dimensions are explicitly enlarged to include those redundant operations calculated by the ILP.
2.  **Sequential State Simulation:** The Python orchestrator runs PyTimeloop on Layer 1, artificially pinning its output mapping to the `GlobalBuffer`. It captures the energy/latency. It then runs Layer 2 (using the enlarged virtual workload) mapped to read from the `GlobalBuffer`.
3.  **Final Tally:** Python aggregates the costs, validating that the analytical ILP model's energy prediction matches the cycle-accurate Timeloop output.

---

## 4. Remaining Engineering Gaps to Smooth Out

While the mathematical core is solid, the transition from theory to a working compiler has specific implementation gaps that must be addressed:

### Gap 1: Complex Topologies (ResNets & Skip Connections)
The zero-search algebraic tiling (Stage 1) is a simple quadratic equation because it models a straight chain (Layer 1 $\rightarrow$ Layer 2). If the network has skip connections (Layer 1 feeds Layer 2 *and* Layer 3), the buffer capacity equation becomes a higher-degree polynomial (Degree 3+). 
*   **The Fix:** We cannot use a simple quadratic formula. We must replace the hardcoded quadratic solver with a fast, deterministic numeric root-finding algorithm (like Newton-Raphson) to instantly find the bounds for higher-degree topologies.

### Gap 2: Data Layout Transformation Penalties
The polyhedral ILP assumes data is stored in abstract, multi-dimensional arrays. However, accelerators are extremely sensitive to physical memory layout (e.g., NHWC vs. NCHW). If the ILP decides the optimal way to compute Layer 2 is to read the data in a different layout than Layer 1 wrote it, there is a hidden energy cost for "transposing" the data in the SRAM.
*   **The Fix:** The ILP objective function must be updated to include a penalty matrix that adds a memory-access cost anytime the affine schedule implies a dynamic layout transformation between consumer and producer.

### Gap 3: NoC Contention in Scale-Out Architectures (Simba)
While we can pin spatial schedules to the PE arrays in the ILP, we cannot easily model the *congestion* of the Network-on-Chip (NoC). If the schedule requires broadcasting weights to 16 chiplets simultaneously, an affine ILP cannot natively model the non-linear latency spikes caused by network collisions.
*   **The Fix:** We cannot solve NoC contention in the ILP. We must rely on Timeloop's detailed NoC analytical models to catch these spikes during Stage 3 (Verification), and potentially add a feedback loop to penalize aggressive multi-cast schedules in the ILP if Timeloop flags a NoC bottleneck.

---

## 5. Theoretical Extension: Multi-Level Memory Hierarchies

Our baseline math assumes a flat 2-level hierarchy (Global Buffer $\rightarrow$ DRAM). Real accelerators possess deeper hierarchies, such as an L1 Scratchpad (local to the PE), an L2 Global Buffer, and L3 DRAM.

### The Decoupled Cascading Solution
Because our analytical Zero-Search Tiling (Stage 1) is perfectly algebraic, extending it to multi-level hierarchies is elegantly simple. We do not need a complex system of simultaneous equations. Instead, we **decouple** the memory levels and solve the quadratic equations in a cascade.

1.  **Solve L2 (Global Buffer):** 
    We run the quadratic formula using `L2_Capacity`. This yields the optimal L2 Tile Size ($T_{L2}$).
2.  **Solve L1 (Scratchpad):** 
    We run the exact same quadratic formula using `L1_Capacity`. This yields the optimal L1 Sub-Tile Size ($T_{L1\_float}$).
3.  **The Perfect Nesting Constraint:** 
    For efficient dataflow, the L1 sub-tiles must divide evenly into the L2 tiles. Instead of rounding $T_{L1\_float}$ to the nearest integer, we mathematically round it down to the largest integer that perfectly divides $T_{L2}$:
    $$ T_{L1\_optimal} = \max \{ d \in \mathbb{N} \mid d \le T_{L1\_float} \land (T_{L2} \bmod d = 0) \} $$

This cascaded root-finding completely solves multi-level spatial tiling analytically in microseconds, ensuring optimal capacity utilization and perfect nesting.
