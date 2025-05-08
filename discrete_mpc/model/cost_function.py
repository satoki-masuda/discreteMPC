import numpy as np


class CostFunction_normal:
    def __init__(self,params):
        self.params = params

    def ttt(self, x):
        num_zones = self.params.num_zones
        N_m = x[:num_zones]
        N_o = x[num_zones : num_zones**2]
        N_p = x[num_zones**2 : ]
        
        # minimize total travel time
        cost = np.sum(N_m) + np.sum(N_o) 
        return cost

class CostFunction_evacuation:
    def __init__(self,params):
        self.params = params

    def tet(self, x):
        
        num_zones = self.params.num_zones
        N_m = x[:num_zones]
        N_s = x[num_zones : 2 * num_zones]
        N_o = x[2 * num_zones : num_zones**2 + num_zones]
        N_d = x[num_zones**2 + 2 * num_zones : ]
        
        # minimize total evacuation time
        tet = np.sum(N_m) + np.sum(N_s) + np.sum(N_o) + np.sum(N_d)
        
        return tet
    
    # terminal cost
    def risk_people(self, x, N_nevac):
        num_zones = self.params.num_zones
        N_m = x[:num_zones]
        N_s = x[num_zones : 2 * num_zones]
        N_o = x[2 * num_zones : num_zones**2 + num_zones]
        N_d = x[num_zones**2 + 2 * num_zones : ]
        
        # minimize the number of people who have not evacuated from flood zones 4,5,6 at the terminal time
        risk_people = np.sum(N_m[4:7]) + np.sum(N_s[4:7]) + np.sum(N_o[4:7]) + np.sum(N_d[4:7]) + np.sum(N_nevac[4:7])
        
        return risk_people