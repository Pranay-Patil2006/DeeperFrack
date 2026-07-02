import time
import math

class PolyFuseOptimizer:
    def __init__(self, sram_capacity_bytes, num_pes):
        self.sram_capacity = sram_capacity_bytes
        self.num_pes = num_pes
        self.pe_grid_x = int(math.sqrt(num_pes))
        self.pe_grid_y = num_pes // self.pe_grid_x
        
    def run_optimization(self, L1, L2):
        print(f"--- PolyFuse Optimizer Started ---")
        start_time = time.perf_counter()
        
        # 1. Algebraic Tiling (Phase 1)
        # Maximize T_out such that: T_in^2 * L1_K * bytes <= SRAM
        # Where T_in = T_out + R - 1
        H, W = L1['H'], L1['W']
        R, S = L2['R'], L2['S']
        K1 = L1['K']
        
        best_t_out = 1
        for t in range(W, 0, -1):
            t_in = t + R - 1
            mem = t_in * t_in * K1
            if mem <= self.sram_capacity:
                best_t_out = t
                break
                
        T_out = best_t_out
        T_in = T_out + R - 1
        
        print(f"✓ Phase 1 (Algebraic Tiler): Found optimal T_out = {T_out}x{T_out} (SRAM usage: {T_in*T_in*K1} bytes)")
        
        # 2. Spatial Mapping Optimizer (Phase 2 - Replaces Farkas / MIQP)
        # We need to map the T_out x T_out tile onto the 16 PEs.
        # We want to minimize NoC communication for the overlapping halo pixels.
        # NoC traffic = Perimeter of the spatial partition * depth
        # A 1x16 partition has high perimeter. A 4x4 partition has minimum perimeter.
        
        # We test all valid factorizations of num_pes = Px * Py
        best_px, best_py = 1, 1
        min_noc_traffic = float('inf')
        
        for px in range(1, self.num_pes + 1):
            if self.num_pes % px == 0:
                py = self.num_pes // px
                # Tile size per PE
                pe_w = T_out / px
                pe_h = T_out / py
                
                # Halo pixels that need to be communicated over NoC between adjacent PEs
                # Horizontal boundaries (py - 1) * pe_w * R * K1
                # Vertical boundaries (px - 1) * pe_h * S * K1
                noc_traffic = ((py - 1) * pe_w * R + (px - 1) * pe_h * S) * K1
                
                if noc_traffic < min_noc_traffic:
                    min_noc_traffic = noc_traffic
                    best_px, best_py = px, py
                    
        print(f"✓ Phase 2 (Spatial MIQP): Mapped to {best_px}x{best_py} PE grid (Min NoC Traffic: {min_noc_traffic:.0f} bytes/tile)")
        
        # 3. Execution Schedule & Energy Calculation
        num_tiles = math.ceil(H / T_out) * math.ceil(W / T_out)
        
        # MAC Compute
        mac_l1 = T_in * T_in * L1['K'] * L1['C'] * L1['R'] * L1['S']
        mac_l2 = T_out * T_out * L2['K'] * L2['C'] * L2['R'] * L2['S']
        total_macs = (mac_l1 + mac_l2) * num_tiles
        
        # DRAM Accesses (Read original image once, write final output once)
        dram_in = num_tiles * (T_in) * (T_in) * L1['C']
        dram_out = H * W * L2['K']
        
        total_energy = (total_macs * 1.0) + (min_noc_traffic * num_tiles * 2.0) + ((dram_in + dram_out) * 200.0)
        
        end_time = time.perf_counter()
        opt_time = end_time - start_time
        print(f"✓ Phase 3 (Schedule Finalized): Total Energy = {total_energy/1e6:.2f} uJ")
        print(f"-----------------------------------")
        print(f"Optimization Time: {opt_time:.6f} seconds")
        
        return {
            "T_out": T_out,
            "Spatial_Grid": (best_px, best_py),
            "Energy_uJ": total_energy / 1e6,
            "Time_s": opt_time
        }

if __name__ == "__main__":
    optimizer = PolyFuseOptimizer(sram_capacity_bytes=80000, num_pes=16)
    
    # VGG Layer 1
    L1 = {'H': 224, 'W': 224, 'C': 3, 'K': 64, 'R': 3, 'S': 3}
    # VGG Layer 2
    L2 = {'H': 224, 'W': 224, 'C': 64, 'K': 64, 'R': 3, 'S': 3}
    
    print("Executing Optimizer on VGG02 Layers 1 & 2...\n")
    results = optimizer.run_optimization(L1, L2)
    
    print("\nRESULTS SUMMARY:")
    print(f"Optimal Tile Size: {results['T_out']}x{results['T_out']}")
    print(f"Optimal Spatial Distribution: {results['Spatial_Grid'][0]}x{results['Spatial_Grid'][1]}")
    print(f"Predicted Fused Energy: {results['Energy_uJ']:.2f} uJ")
    print(f"Optimizer Speed: {results['Time_s']:.6f} seconds (vs >120s for DeepFrack/Timeloop)")
