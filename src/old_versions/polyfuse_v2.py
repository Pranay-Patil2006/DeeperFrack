import time
import math

class PolyFuseOptimizerV2:
    def __init__(self, sram_capacity_bytes, num_pes):
        self.sram_capacity = sram_capacity_bytes
        self.num_pes = num_pes
        
        # Energy Costs (pJ)
        self.COST_MAC = 1.0
        self.COST_NOC = 2.0    # Per byte hop
        self.COST_DRAM = 200.0 # Per byte

    def run_optimization(self, L1, L2):
        print(f"--- PolyFuse Optimizer V2 (Includes Weight Caching & Dataflow) ---")
        start_time = time.perf_counter()
        
        H, W = L1['H'], L1['W']
        R1, S1, K1, C1 = L1['R'], L1['S'], L1['K'], L1['C']
        R2, S2, K2, C2 = L2['R'], L2['S'], L2['K'], L2['C']
        
        # Weight volumes (bytes, assuming 1 byte per weight for simplicity of ratio)
        W1_vol = R1 * S1 * C1 * K1
        W2_vol = R2 * S2 * C2 * K2
        
        best_energy = float('inf')
        best_config = {}

        # Explore Weight Caching Policies
        for cache_w1 in [True, False]:
            for cache_w2 in [True, False]:
                
                # 1. Algebraic Tiling (Phase 1)
                # Available SRAM for Activations
                avail_sram = self.sram_capacity
                if cache_w1: avail_sram -= W1_vol
                if cache_w2: avail_sram -= W2_vol
                
                if avail_sram <= 0: continue
                
                best_t = 1
                for t in range(W, 0, -1):
                    t_in = t + R2 - 1
                    activations_mem = t_in * t_in * K1
                    if activations_mem <= avail_sram:
                        best_t = t
                        break
                        
                T_out = best_t
                T_in = T_out + R2 - 1
                num_tiles = math.ceil(H / T_out) * math.ceil(W / T_out)
                
                # 2. Spatial Mapping (Phase 2 - NoC Optimization)
                best_px, best_py = 1, 1
                min_noc_traffic = float('inf')
                for px in range(1, self.num_pes + 1):
                    if self.num_pes % px == 0:
                        py = self.num_pes // px
                        pe_w = T_out / px
                        pe_h = T_out / py
                        noc_traffic = ((py - 1) * pe_w * R2 + (px - 1) * pe_h * S2) * K1
                        if noc_traffic < min_noc_traffic:
                            min_noc_traffic = noc_traffic
                            best_px, best_py = px, py
                            
                # 3. Exact Energy Calculation
                # Compute (MACs)
                mac_l1 = T_in * T_in * K1 * C1 * R1 * S1
                mac_l2 = T_out * T_out * K2 * C2 * R2 * S2
                total_macs = (mac_l1 + mac_l2) * num_tiles
                
                # DRAM Accesses
                # Inputs: read from DRAM for every tile (T_in x T_in x C1)
                dram_inputs = num_tiles * (T_in * T_in * C1)
                
                # Outputs: written to DRAM once
                dram_outputs = H * W * K2
                
                # Weights: if cached, read once. If not, read per tile.
                dram_w1 = W1_vol if cache_w1 else (W1_vol * num_tiles)
                dram_w2 = W2_vol if cache_w2 else (W2_vol * num_tiles)
                
                total_dram_bytes = dram_inputs + dram_outputs + dram_w1 + dram_w2
                
                # Total Energy (in pJ, then convert to uJ)
                energy_compute = total_macs * self.COST_MAC
                energy_noc = (min_noc_traffic * num_tiles) * self.COST_NOC
                energy_dram = total_dram_bytes * self.COST_DRAM
                
                total_energy_uJ = (energy_compute + energy_noc + energy_dram) / 1e6
                
                if total_energy_uJ < best_energy:
                    best_energy = total_energy_uJ
                    best_config = {
                        "T_out": T_out,
                        "Cache_W1": cache_w1,
                        "Cache_W2": cache_w2,
                        "Grid": f"{best_px}x{best_py}",
                        "Energy_uJ": total_energy_uJ,
                        "DRAM_Bytes": total_dram_bytes,
                        "Compute_Energy_uJ": energy_compute / 1e6,
                        "DRAM_Energy_uJ": energy_dram / 1e6
                    }
                    
        end_time = time.perf_counter()
        opt_time = end_time - start_time
        best_config["Opt_Time_s"] = opt_time
        return best_config

if __name__ == "__main__":
    # Baseline Isolated Layer Energy Estimator
    def get_isolated_energy(L1, L2, H, W):
        # Layer 1 reads input + W1 from DRAM, writes intermediate to DRAM
        dram_in1 = H * W * L1['C']
        dram_w1 = L1['R'] * L1['S'] * L1['C'] * L1['K']
        dram_out1 = H * W * L1['K']
        macs1 = H * W * L1['K'] * L1['C'] * L1['R'] * L1['S']
        
        # Layer 2 reads intermediate + W2 from DRAM, writes output to DRAM
        dram_in2 = H * W * L2['C']
        dram_w2 = L2['R'] * L2['S'] * L2['C'] * L2['K']
        dram_out2 = H * W * L2['K']
        macs2 = H * W * L2['K'] * L2['C'] * L2['R'] * L2['S']
        
        total_dram = dram_in1 + dram_w1 + dram_out1 + dram_in2 + dram_w2 + dram_out2
        total_macs = macs1 + macs2
        
        # 200 pJ per DRAM byte, 1 pJ per MAC
        return ((total_dram * 200.0) + (total_macs * 1.0)) / 1e6


    optimizer = PolyFuseOptimizerV2(sram_capacity_bytes=80000, num_pes=16)
    
    L1 = {'H': 224, 'W': 224, 'C': 3, 'K': 64, 'R': 3, 'S': 3}
    L2 = {'H': 224, 'W': 224, 'C': 64, 'K': 64, 'R': 3, 'S': 3}
    
    res = optimizer.run_optimization(L1, L2)
    iso_energy = get_isolated_energy(L1, L2, 224, 224)
    
    print("\nRESULTS SUMMARY:")
    print(f"Optimal Tile Size: {res['T_out']}x{res['T_out']}")
    print(f"Weight Caching Policy: Layer1_Pinned={res['Cache_W1']}, Layer2_Pinned={res['Cache_W2']}")
    print(f"Spatial Grid: {res['Grid']}")
    print(f"Optimizer Execution Time: {res['Opt_Time_s']:.6f} seconds")
    
    print("\nENERGY COMPARISON:")
    print(f"Isolated System Energy: {iso_energy:.2f} uJ")
    print(f"PolyFuse V2 System Energy: {res['Energy_uJ']:.2f} uJ")
    print(f"Energy Reduction: {((iso_energy - res['Energy_uJ']) / iso_energy * 100):.2f}%")
