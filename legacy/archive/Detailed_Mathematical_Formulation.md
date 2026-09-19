# Detailed Mathematical Formulation: Analytical CNN Fusion

This report provides the rigorous, step-by-step mathematical foundation for the Analytical CNN Fusion System. We leave no variables to assumption. 

*Note: For edge/inference accelerators, we assume Batch Size $N=1$. It has been removed to simplify the formulations.*

---

## 1. Workload Notation & Polyhedral Representation

Let us formally define a 2D Convolution operation for a Layer $i$. 
We use the standard deep learning loop bounds:
*   $K$: Output channels
*   $C$: Input channels
*   $H_{out}, W_{out}$: Output spatial dimensions
*   $R, S$: Filter height and width (Kernel Size)
*   $U, V$: Stride in height and width

### A. The Iteration Domain ($\mathcal{D}$)
The iteration domain represents every single MAC operation as an integer coordinate vector $\vec{x}$ in a 6-dimensional space (batch size removed).
$$\mathcal{D}_i = \{ (k, c, h, w, r, s) \in \mathbb{Z}^6 \mid 0 \le k < K, \dots \}$$

### B. Access Relations ($\mathcal{A}$)
An access relation maps an iteration point $\vec{x}$ to a specific memory address in a tensor.
For Layer $i$:
*   **Output Tensor Access ($O_i$):** $(k, h, w)$
*   **Weight Tensor Access ($W_i$):** $(k, c, r, s)$
*   **Input Tensor Access ($I_i$):** $(c, h_{in}, w_{in})$ 
    *   Where the halo mapping is strictly defined by:
    *   $h_{in} = h \cdot U + r$
    *   $w_{in} = w \cdot V + s$

### C. The Fusion Dependence ($\mathcal{P}$)
When fusing Layer 1 $\rightarrow$ Layer 2, the output of Layer 1 is the input to Layer 2 ($O_1 = I_2$). 
The mathematical dependence between a consumer point $\vec{x}_2$ and a producer point $\vec{x}_1$ is defined when their access relations intersect:
$$\mathcal{P}_{1 \rightarrow 2} = \{ (\vec{x}_1, \vec{x}_2) \mid O_1(\vec{x}_1) = I_2(\vec{x}_2) \}$$

---

## 2. Stage 1: Zero-Search Analytical Tiling ("Pyramid Tiling")

### A. PE Quantization vs. SRAM Underutilization
While the absolute maximum spatial tile size minimizes DRAM bandwidth, it ignores **PE Array Quantization**. If the max tile size is $21$, but the PE array is $16 \times 16$, the remainder block ($5 \times 5$) leaves the majority of the PEs idle.

If we mathematically round down the tile size to the nearest PE multiple (e.g., $16$), we guarantee 100% PE utilization, but we create **SRAM Underutilization** (slack space).
*   **The Optimization:** This slack space does not go to waste! The algebraic solver detects this leftover SRAM capacity and dynamically reallocates it to increase the Channel Tile dimensions ($T_k$ or $T_c$), or it naturally absorbs the expanding intermediate "halo" data for deeper fusions. We perfectly balance the SRAM and PE arrays without iterative search.

### B. Formulating the Footprint (Peak vs. Persistent Intermediate Data)
To compute the output tile, the *entire* intermediate input tile must exist in SRAM at the exact moment the PEs are firing. This is the **Peak Active Footprint**.
*   **Output Footprint ($Vol_{out}$):** $T_k \cdot T_{out}^2$
*   **Input Footprint ($Vol_{in}$):** $T_c \cdot (T_{out} + R - 1)^2$
*   **Weight Footprint ($Vol_{weights}$):** $T_k \cdot T_c \cdot R^2$

If we choose $\delta=0$ (we decide to store the overlapping halo for the *next* adjacent tile instead of recalculating), that halo must persist in memory.
*   **Persistent Halo ($Vol_{halo}$):** $T_c \cdot ( (T_{out} + R - 1)^2 - T_{out}^2 )$

**The Exact Buffer Constraint Inequality:**
$$ Peak\_Active\_Footprint + Persistent\_Halo \le SRAM_{cap} $$
$$ (Vol_{out} + Vol_{in} + Vol_{weights}) + (Vol_{halo} \cdot (1 - \delta)) \le SRAM_{cap} $$

---

## 3. Stage 2: Formal ILP Scheduling (Hazards & Causality)

Once the tile sizes are fixed, the Integer Linear Programming (ILP) solver finds the optimal execution order (the Schedule $\Theta$).

### A. Preventing Dependency Hazards (Farkas' Lemma)
For Layer $i$, the affine schedule assigns a multidimensional timestamp to every iteration point $\vec{x}$.
$$ t_i(\vec{x}) = \Theta_i \cdot \vec{x} + \vec{p}_i $$

To guarantee no hazards, the ILP enforces **Strict Causality**. A consumer point $\vec{x}_2$ in Layer 2 cannot be scheduled before the producer point $\vec{x}_1$ in Layer 1.
For every single dependence vector in $\mathcal{P}_{1 \rightarrow 2}$, the ILP enforces the inequality:
$$ \Theta_2 \cdot \vec{x}_2 - \Theta_1 \cdot \vec{x}_1 \ge 1 $$

The ILP uses **Farkas' Lemma** to convert this infinite set of dependence points into a finite set of affine constraints, mathematically proving the schedule is hazard-free.

### B. The Objective Function (Uniform Fusion Boundaries)
The decision to recalculate vs. store is made *once per fused layer boundary*. Let $\delta_{(1 \rightarrow 2)}$ be a binary decision variable that applies to the entire loop fusion.

**Minimize Total Energy:**
$$ E_{total} = (Cost_{MAC} \cdot Vol_{overlap} \cdot \delta_{(1 \rightarrow 2)}) + (Cost_{SRAM} \cdot Vol_{halo} \cdot (1 - \delta_{(1 \rightarrow 2)})) $$

### C. Constraint: Hardware Spatial Pinning
For all points $\vec{x} \in \mathcal{D}$, the spatial execution must fit the physical PE dimensions:
$$ 0 \le \Theta^{SpaceX} \cdot \vec{x} < PE_w $$
$$ 0 \le \Theta^{SpaceY} \cdot \vec{x} < PE_h $$

---

## 4. System Interfaces: Data Extraction and Energy Modeling

To build this system on top of Timeloop, we must extract specific configurations and handle complex energy actions.

### A. Extracting Information from YAML Inputs
The polyhedral compiler acts as a bridge, pulling the following explicit data to construct the mathematical bounds:
*   **From `arch.yaml`:**
    *   `SRAM_Capacity` and `DRAM_Capacity` (to bound the footprint equations).
    *   `PE_Array_Width` and `PE_Array_Height` (to set spatial constraints $PE_w, PE_h$).
    *   Baseline Energy Costs (e.g., Joules per SRAM read, Joules per MAC).
*   **From `prob.yaml`:**
    *   The complete CNN topology: Loop bounds ($K, C, H, W$), Kernel sizes ($R, S$), and strides ($U, V$) to construct the Iteration Domains ($\mathcal{D}$).

### B. Handling Multiple Actions and Different Energies
In reality, a "MAC" or a "Memory Access" in hardware consists of multiple sub-actions (e.g., Register File reads, ALU compute, spatial multicast via NoC vs. unicast). Each of these has a vastly different energy cost. 

If we try to encode every micro-architectural action into the ILP objective function, the mathematical model will become too complex to solve.
*   **The Proxy Solution:** The ILP Objective Function uses an **Aggregated Energy Proxy**. We extract a blended average cost for $Cost_{MAC}$ and $Cost_{SRAM}$ from the `arch.yaml`. This proxy is mathematically sufficient to make the structural decision of $\delta$ (Recalculate vs. Store). 
*   **Timeloop as Ground Truth:** Once the ILP uses the proxies to find the optimal Looptree schedule, we pass it to Timeloop. Timeloop executes the schedule and calculates the exact, micro-architectural energy breakdown for every distinct sub-action (multicast, ALU, RF read). This elegantly separates structural optimization from cycle-accurate energy accounting.
