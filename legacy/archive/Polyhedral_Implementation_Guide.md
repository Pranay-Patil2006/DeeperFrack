# Polyhedral Implementation Guide: From Theory to Code

You are completely right to call out the "hand-wavy" nature of simply saying "the ILP solves it." Building a custom polyhedral compiler is notoriously difficult. 

To prove this is implementable and to remove all ambiguity, this document details the exact software architecture, libraries, and Python-level formulations required to build the Polyhedral Mapping Engine using **`islpy`** (the Python wrapper for the Integer Set Library).

---

## 1. The Core Toolchain

We will not build an ILP solver from scratch. We will use the industry standard mathematical engine that powers Pluto and LLVM's Polly: **ISL (Integer Set Library)** via `islpy`. 

The implementation workflow consists of 4 concrete software steps:
1.  **YAML to ISL String Translation**
2.  **Domain and Access Map Construction**
3.  **Dependence Calculation (The Engine)**
4.  **Schedule Generation via ISL ILP**

---

## 2. Concrete Implementation Steps

### Step 1 & 2: Translating YAML to ISL Sets and Maps
We read `prob.yaml` and convert the loop bounds into strict ISL mathematical strings.

```python
import islpy as isl

# Initialize the mathematical context
ctx = isl.Context()

# Define the parametric space from prob.yaml
# K=OutChan, C=InChan, H/W=Spatial, R/S=Kernel
params = "[K, C, H, W, R, S]"

# 1. Construct Iteration Domains for Layer 1 and Layer 2
# S1 is a point in Layer 1; S2 is a point in Layer 2
S1_domain = isl.BasicSet(f"{params} -> {{ S1[k, c, h, w, r, s] : 0<=k<K and 0<=c<C and 0<=h<H and 0<=w<W and 0<=r<R and 0<=s<S }}")
S2_domain = isl.BasicSet(f"{params} -> {{ S2[k2, c2, h2, w2, r2, s2] : 0<=k2<K and 0<=c2<K and 0<=h2<H and ... }}")

# 2. Construct Access Relations (Memory mapping)
# Layer 1 Writes to Tensor O1
S1_write_O1 = isl.BasicMap(f"{params} -> {{ S1[k, c, h, w, r, s] -> O1[k, h, w] }}")

# Layer 2 Reads from Tensor O1 (Notice the halo mapping h2+r2)
S2_read_O1 = isl.BasicMap(f"{params} -> {{ S2[k2, c2, h2, w2, r2, s2] -> O1[c2, h2+r2, w2+s2] }}")
```

### Step 3: Exact Dependence Calculation
We do not manually calculate the "halo" overlap. We ask ISL to mathematically compute the exact flow dependencies between Layer 1 and Layer 2.

```python
# To find exactly which S1 point produces the data that S2 needs:
# We intersect the Write Map of S1 with the Read Map of S2
dependences = S1_write_O1.apply_range(S2_read_O1.reverse())

# The 'dependences' object is now a mathematically exact Polyhedron defining 
# P_{1->2}. It tells the solver exactly what must execute before what.
```

### Step 4: The Custom Hardware Cost Function (Bypassing ISL's CPU Scheduler)

*You correctly identified a massive flaw in standard polyhedral compilers.* The built-in `isl.compute_schedule()` uses the Pluto objective function. The Pluto objective function assumes a CPU/Cache architecture and strictly minimizes the iteration distance between producer and consumer. It has zero understanding of spatial PE arrays, SRAM vs. DRAM energy costs, or Dataflows (like Row-Stationary).

If we use ISL's native scheduler, it will output a schedule optimized for an Intel CPU, not Eyeriss.

**The Fix: The Hybrid ISL-Gurobi Solver**
To implement our custom accelerator cost function, we must decouple the dependence math from the ILP solver.

1.  **Extract the Constraints (ISL):** We use ISL purely as a mathematical engine to apply Farkas' Lemma to our `dependences` map. This converts the infinite polyhedral points into a finite set of linear inequality equations (guaranteeing strict causality).
2.  **Export to Python ILP (Gurobi/PuLP):** We export these equations from ISL into a dedicated Python mathematical solver (like Gurobi).
3.  **Define the Custom Objective (Gurobi):** Inside Gurobi, we write our custom objective function based on `arch.yaml`:
    ```python
    # Inside Gurobi Python API
    # Objective: Minimize Energy (Cost_MAC * Recompute + Cost_DRAM * Spill)
    m.setObjective(Cost_MAC * Vol_overlap_var + Cost_DRAM * Vol_spill_var, GRB.MINIMIZE)
    
    # Add Hardware Pinning Constraints
    m.addConstr(Theta_SpaceX <= PE_w)
    m.addConstr(Theta_SpaceY <= PE_h)
    
    # Add Farkas' Lemma Legality Constraints extracted from ISL
    m.addConstrs(...) 
    
    m.optimize()
    ```
4.  **Inject Back to ISL:** Gurobi solves the ILP and outputs the optimal schedule coefficients ($\Theta$). We read these coefficients and inject them *back* into ISL as a custom `isl.Map` or `isl.Schedule`.
5.  **Output to Timeloop:** We use ISL's code generator to output the Timeloop `map.yaml` from our custom, hardware-optimized schedule.

---

## 3. Resolving the "Recalculate vs Store" ILP Decision 

In the previous theory, I mentioned using a binary variable $\delta$ in the ILP objective function to decide whether to recalculate the halo. 

Because we separated the Tile Size Calculation (Stage 1) from the ILP (Stage 2), our Python orchestrator makes the $\delta$ decision *before* generating the constraints:
1.  Python calculates the $Vol_{halo}$.
2.  Python evaluates: `if (Vol_halo * Cost_MAC) < (Vol_spill * Cost_DRAM)`
3.  If True (Recalculate): Python uses a polyhedral trick called *Index Set Splitting*. It literally duplicates the `isl.BasicSet` for Layer 1 for every tile, completely severing the mathematical dependency between adjacent tiles, forcing the solver to schedule them to recalculate the halo independently.
4.  If False (Store): Python passes the normal `isl.BasicSet` to the solver, and it schedules them to share the halo data sequentially through SRAM.
