import math

class AnalyticalTiler:
    def __init__(self, buffer_in, buffer_out, buffer_weight, pe_array_dim):
        """
        Initializes the Stage 1 Algebraic solver with hardware constraints.
        Using separated buffers to ensure easy FSM control logic.
        """
        self.buf_in = buffer_in
        self.buf_out = buffer_out
        self.buf_weight = buffer_weight
        self.pe_dim = pe_array_dim

    def calculate_optimal_tiles(self, R, S):
        """
        Analytically solves for the optimal spatial (T_out) and channel (T_k, T_c) 
        tile sizes to maximize the utilization of separated SRAM buffers.
        
        R, S: Filter height and width (e.g., 3x3)
        """
        # Step 1: Maximize Channel Tiles based on Weight Buffer
        # T_k * T_c * R * S <= Buf_Weight
        # For a balanced dataflow, we assume T_k ~= T_c. Let T_chan = T_k = T_c.
        # T_chan^2 * R * S = Buf_Weight  =>  T_chan = sqrt(Buf_Weight / (R*S))
        max_t_chan_float = math.sqrt(self.buf_weight / (R * S))
        
        # Round down to the nearest multiple of the PE array dimension to ensure no idle PEs
        T_chan_optimal = (math.floor(max_t_chan_float / self.pe_dim)) * self.pe_dim
        
        # If the buffer is smaller than PE dim, just floor it.
        if T_chan_optimal == 0:
            T_chan_optimal = math.floor(max_t_chan_float)
            
        T_k = T_chan_optimal
        T_c = T_chan_optimal

        # Step 2: Maximize Output Spatial Tile based on Output Buffer
        # T_k * T_out^2 <= Buf_Out  =>  T_out = sqrt(Buf_Out / T_k)
        if T_k > 0:
            max_t_out_float = math.sqrt(self.buf_out / T_k)
            T_out_optimal_out = (math.floor(max_t_out_float / self.pe_dim)) * self.pe_dim
            if T_out_optimal_out == 0: T_out_optimal_out = math.floor(max_t_out_float)
        else:
            T_out_optimal_out = 0

        # Step 3: Maximize Input Spatial Tile based on Input Buffer
        # T_c * (T_out + R - 1) * (T_out + S - 1) <= Buf_In
        # Assuming R=S for square kernels, T_c * (T_out + R - 1)^2 <= Buf_In
        if T_c > 0:
            # (T_out + R - 1) = sqrt(Buf_In / T_c)
            max_t_out_in_float = math.sqrt(self.buf_in / T_c) - R + 1
            T_out_optimal_in = (math.floor(max_t_out_in_float / self.pe_dim)) * self.pe_dim
            if T_out_optimal_in == 0: T_out_optimal_in = math.floor(max_t_out_in_float)
        else:
            T_out_optimal_in = 0

        # The limiting factor is the strictest spatial constraint
        T_out_final = min(T_out_optimal_out, T_out_optimal_in)
        
        # Calculate resulting T_in
        T_in_final = T_out_final + R - 1

        return {
            "T_k": T_k,
            "T_c": T_c,
            "T_out": T_out_final,
            "T_in": T_in_final,
            "Vol_Weight": T_k * T_c * R * S,
            "Vol_Out": T_k * (T_out_final**2),
            "Vol_In": T_c * (T_in_final**2)
        }

if __name__ == "__main__":
    # Example Hardware (e.g., Eyeriss-like separated buffers)
    # 100KB In, 100KB Out, 50KB Weights
    solver = AnalyticalTiler(buffer_in=100000, buffer_out=100000, buffer_weight=50000, pe_array_dim=16)
    
    # Example Layer (3x3 Convolution)
    tiles = solver.calculate_optimal_tiles(R=3, S=3)
    
    print("=== Stage 1: Zero-Search Analytical Tiling ===")
    print(f"Optimal Output Channels (T_k): {tiles['T_k']}")
    print(f"Optimal Input Channels  (T_c): {tiles['T_c']}")
    print(f"Optimal Output Spatial (T_out): {tiles['T_out']} x {tiles['T_out']}")
    print(f"Required Input Spatial  (T_in): {tiles['T_in']} x {tiles['T_in']}")
    print("-" * 40)
    print("Buffer Utilizations:")
    print(f"Weights: {tiles['Vol_Weight']} / 50000")
    print(f"Output:  {tiles['Vol_Out']} / 100000")
    print(f"Input:   {tiles['Vol_In']} / 100000")
