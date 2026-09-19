# PolyFuse V2: Analytical CNN Layer Fusion Optimizer

## Introduction
In the realm of Deep Neural Network (DNN) hardware acceleration, data movement (especially off-chip DRAM access) is often the dominant source of energy consumption and latency. Layer fusion is a technique that keeps intermediate feature maps on-chip between consecutive CNN layers, avoiding expensive round-trips to main memory. 

**PolyFuse V2** is an advanced **Analytical CNN Layer Fusion Optimizer**. Unlike naive layer-by-layer execution or greedy fuse-all heuristics, PolyFuse V2 intelligently determines the *globally optimal* partitioning of a sequence of CNN layers into fused execution blocks to minimize a target objective (Energy, Latency, or Energy-Delay Product).

---

## Core Philosophy
PolyFuse V2 guarantees a scientifically valid optimization by explicitly modeling the underlying hardware architecture. It avoids expensive exhaustive search spaces by leveraging analytical formulas for tiling and routing, subsequently using Dynamic Programming (DP) to find the mathematically optimal layer grouping.

---

## The PolyFuse V2 Algorithm

The algorithm is structured into three consecutive analytical stages:

### Stage 1: Zero-Search Analytical Tiling
The first stage determines whether a proposed stack of layers can be feasibly fused within the constrained memory hierarchy of the target accelerator. 

**Mechanism:**
Rather than relying on iterative search, it computes the maximum valid tile sizes for the fused layers using a **polyhedral recurrence formula**:
`T_in^(i) = (T_out^(i) - 1) * stride^(i) + R^(i)`

Where:
*   `T_in^(i)` = Input tile size for layer `i`
*   `T_out^(i)` = Output tile size for layer `i`
*   `stride^(i)` = Convolutional stride
*   `R^(i)` = Kernel size

**Validation:**
The tiler dynamically reads the Hardware Abstraction Layer (HAL) architecture parameters (e.g., Global Shared Buffer, PE-local SRAM capacities) and restricts the maximum output tile size `T_out` so that the combined footprint of weights, inputs, and intermediate activations strictly fits into the on-chip buffers. If it overflows, the fusion is marked mathematically infeasible.

### Stage 2: Analytical Spatial Routing
For layers that are fused and mapped to a spatial 2D array of Processing Elements (PEs), data must flow efficiently between PEs to avoid Network-on-Chip (NoC) congestion.

**Mechanism:**
This stage optimizes the physical spatial allocation `(X, Y)` of operations on the PE mesh array. 
*   **Objective:** It minimizes the physical NoC hop distances by penalizing the sum of squared distances `(X2 - X1)^2 + (Y2 - Y1)^2` between dependent consecutive layers.
*   **Constraints:** It applies extreme vector causality bounds to ensure that data dependencies flow forward spatially without violating scheduling constraints. 

### Stage 3: Dynamic Programming (DP) Partitioner (In-Depth)
The final stage determines the absolute best way to cut the entire CNN model into a series of fused blocks. Unlike greedy heuristics which simply attempt to fuse as many layers as possible until memory overflows, the DP partitioner mathematically guarantees the global optimum for the chosen objective (Energy, Latency, or Energy-Delay Product).

**Mathematical Formulation:**
Let $N$ be the total number of layers in the CNN. 
Let $DP[i]$ represent the minimum cumulative cost to execute the first $i$ layers of the network. 
The DP array is initialized such that $DP[0] = 0$ and $DP[1 \dots N] = \infty$.

The state transition evaluates every possible sub-sequence (or "stack") of layers from index $j$ to $i$ (where $0 \le j < i$). The recurrence relation is defined as:

$$DP[i] = \min_{0 \le j < i} \Big( DP[j] + \text{TotalCost}(layers[j:i]) \Big)$$

**Cost Evaluation Workflow (`TotalCost`):**
For each candidate block of layers from $j$ to $i$, the algorithm calculates the cost through a multi-step verification process:

1.  **Fast Feasibility Check (Stage 1):** Before invoking expensive hardware simulations, the DP solver passes the stack $layers[j:i]$ to the Stage 1 Tiler. If the tiler determines that the required tile sizes exceed the physical capacities of the on-chip SRAM/Global Buffers, the cost of this block evaluates to $\infty$, and the DP transitions to the next candidate.
2.  **Base Hardware Cost Evaluation:** If the block fits in memory, the DP invokes `LoopTree` (a proxy evaluator backed by Timeloop/Accelergy) to accurately simulate the hardware behavior and retrieve the base hardware cost (e.g., base energy in picojoules).
3.  **Spatial Routing Penalty (Stage 2):** The DP invokes the Stage 2 Router to compute the physical NoC routing cost for mapping the dependent layers across the 2D PE mesh. This is calculated as a penalty factor based on the sum of squared distances: 
    $\text{RoutingCost} = \sum ((X_{k+1} - X_k)^2 + (Y_{k+1} - Y_k)^2) \times 5.0$
4.  **Objective Scaling:** The final `TotalCost` is adjusted based on the user's optimization target:
    *   **Energy:** $\text{TotalCost} = \text{BaseEnergy} + \text{RoutingCost}$
    *   **Latency:** $\text{TotalCost} = (\text{BaseEnergy} + \text{RoutingCost}) / 100.0$ *(using a heuristic proxy scaling factor)*
    *   **EDP:** $\text{TotalCost} = (\text{BaseEnergy} + \text{RoutingCost}) \times \text{Latency}$

**Reconstruction:**
During the forward pass, the algorithm maintains a `parent[i]` array to store the optimal split point $j$ that yielded the minimum cost for $DP[i]$. It also caches metadata (`best_meta[i]`) including the optimal tile sizes, weight caching patterns, and PE routing assignments for that block. 

Once $DP[N]$ is computed, the algorithm backtracks through the `parent` array starting from $N$ down to $0$. This reconstruction yields the exact sequence of optimal partitions and all the hardware mapping metadata required to deploy the fused model onto the accelerator.

---

## Comparison With Baselines

In academic evaluations, PolyFuse V2 is typically evaluated against two standard paradigms on the exact same hardware architecture:
1.  **Naive Baseline (Layer-by-Layer):** Executes one layer at a time, writing all intermediate activations back to DRAM. PolyFuse V2 consistently outperforms this by eliminating off-chip intermediate traffic.
2.  **Greedy Fuse-All Baseline (Implemented in codebase as "DeepFrack Baseline"):** Attempts to greedily fuse all layers into one massive block. If the network is too deep, the intermediate feature maps overflow the on-chip Global Buffer, rendering the mapping physically infeasible. PolyFuse V2 avoids this by strategically breaking the fusion chain at optimal points. *(Note: While the PolyFuse V2 testbed labels this greedy heuristic as "DeepFrack", the actual DeepFrack framework is a sophisticated tool that also utilizes dynamic programming to optimally partition and tile workloads.)*
