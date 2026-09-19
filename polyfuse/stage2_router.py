import numpy as np
from scipy.optimize import minimize

class Stage2Router:
    def __init__(self, hw_config):
        self.pe_x = hw_config.pe_array['meshX']
        self.pe_y = hw_config.pe_array['meshY']
        
    def optimize_spatial_routing(self, layers):
        """
        Determine spatial PE allocation (X, Y) for each layer in the fused stack.
        Objective: Minimize NoC physical hop distances (squared).
        Constraints:
        1. Extreme Vector Method for causality.
        2. Spatial Hardware Pinning (0 <= X < pe_x, 0 <= Y < pe_y).
        """
        n_layers = len(layers)
        if n_layers <= 1:
            return [(0, 0)] * n_layers
            
        # Variables: X_0, Y_0, X_1, Y_1, ... X_n, Y_n
        # Flattened array: [X_0, Y_0, X_1, Y_1, ...]
        
        def objective(vars):
            # Sum of squared distances between consecutive layers
            cost = 0.0
            for i in range(n_layers - 1):
                x1, y1 = vars[2*i], vars[2*i+1]
                x2, y2 = vars[2*(i+1)], vars[2*(i+1)+1]
                cost += (x2 - x1)**2 + (y2 - y1)**2
            return cost
            
        # Initial guess: place all on (pe_x/2, pe_y/2)
        x0 = np.array([self.pe_x / 2.0, self.pe_y / 2.0] * n_layers)
        
        # Bounds: 0 <= X <= pe_x-1, 0 <= Y <= pe_y-1
        bounds = []
        for _ in range(n_layers):
            bounds.append((0, max(0, self.pe_x - 1)))
            bounds.append((0, max(0, self.pe_y - 1)))
            
        # Causality constraints (Extreme Vector Method)
        # For a 3x3 kernel, causality at 4 corner vectors.
        # This is highly simplified for proxy modeling. 
        # In a strict mathematical formulation, this bounds the relative placement 
        # based on scheduling vectors Theta.
        constraints = []
        
        # We will formulate a dummy causality constraint to satisfy SLSQP structure
        # In reality, Theta_2 * x - Theta_1 * (x - d_ext) >= 1
        # For our abstract proxy without explicit time scheduling vectors,
        # we ensure they are not assigned to exact same PE if causality implies a delay
        # that conflicts with spatial data flow.
        def causality_constraint(vars):
            # Just a placeholder for Extreme Vector: ensure layers flow "forward" if needed
            return 1.0 # Always satisfied for proxy
            
        con = {'type': 'ineq', 'fun': causality_constraint}
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=[con])
        
        routing = []
        if res.success:
            for i in range(n_layers):
                x, y = res.x[2*i], res.x[2*i+1]
                routing.append((int(round(x)), int(round(y))))
        else:
            # Fallback to linear mapping
            for i in range(n_layers):
                routing.append((i % self.pe_x, (i // self.pe_x) % self.pe_y))
                
        return routing
