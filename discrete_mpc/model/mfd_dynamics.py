import numpy as np
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import imageio
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm

from parameters_ndp import Parameters
from cost_function import CostFunction_normal, CostFunction_evacuation

np.random.seed(42)

def mfd(N: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray):
    return A * (N**3) + B * (N**2) + C * N

def optimized_k_shortest_theta(N, path_dict, L_m, num_zones, A: float, B: float, C: float, max_boundary_capacity):
    avg_speed = np.divide(mfd(N, A, B, C), N, where=(N!=0))
    theta = np.zeros((num_zones, num_zones, num_zones))
    
    for k in range(num_zones):
        for i in range(num_zones):
            if i != k:
                k_shortest_path = path_dict[(i, k)]
                for path, _ in k_shortest_path:
                    j = path[1]  # next zone after i in the k-th shortest path from i to k
                    if max_boundary_capacity[i, j] > 0:
                        cost = sum([L_m[zone] / avg_speed[zone] if avg_speed[zone] != 0 else 60 for zone in path]) # 0で割るとinfになるので60分にしている
                        theta[i, j, k] += np.exp(-cost)
    
    row_sums = np.sum(theta, axis=1, keepdims=True)
    np.divide(theta, row_sums, out=theta, where=row_sums != 0)
    
    return theta

def redistribute_traffic(N_s, excess_zone, L_o):
    """
    Redistribute excess traffic from congested zones to other zones based on distance weights.
    
    Args:
        N_s (np.ndarray): Traffic accumulation in each zone
        excess_zone (list): List of congested zones to redistribute traffic from
        L_o (np.ndarray): Distance matrix between zones
        
    Returns:
        np.ndarray: Matrix of redistributed traffic flows between zones
    """
    num_zones = len(N_s)
    
    distance_weight = np.divide(1, L_o, where=(L_o != 0))
    # Prohibit movement to excess_zone
    distance_weight[:, excess_zone] = 0
    # Normalize as probability distribution for each zone
    distance_weight /= distance_weight.sum(axis=1, keepdims=True)
    N_re = np.zeros((num_zones, num_zones))
    N_re[excess_zone] = N_s[excess_zone][:,np.newaxis] * distance_weight[excess_zone,:]
    
    return N_re

class MFD_Dynamics:
    """
    MFD_Dynamics class simulates traffic dynamics based on Macroscopic Fundamental Diagram (MFD).
    
    This class handles:
    - Traffic flow simulation using MFD
    - State updates for both evacuating and background traffic
    - Cost calculations for travel time and evacuation time
    - Traffic redistribution when zones are congested
    
    Args:
        params: Instance of Parameters class containing simulation parameters
        output_path: Path to save simulation outputs (default: None)
        
    Attributes:
        x: State vector for evacuating traffic
        x_background: State vector for background traffic
        u: Control input vector
        ttt: Total travel time
        tet: Total evacuation time
        Q: Origin-destination demand matrix for evacuating traffic
        Q_background: Origin-destination demand matrix for background traffic
    """
    def __init__(self, params, output_path=None):
        self.output_path = output_path
        if self.output_path is not None and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.params = params
        self.sampling_time = self.params.sampling_time 
        self.sim_start_step = self.params.simulation_start_time//self.sampling_time
        self.sim_end_step = self.params.simulation_end_time//self.sampling_time
        self.sim_steps = self.sim_end_step - self.sim_start_step
        self.pos = {0: (1,5.5), 1: (3,2), 2: (5.5,2), 3:(8,4), 4:(2.5,5), 5:(6,4.5), 6:(3.5, 4.25), 7:(3, 7.5), 8:(6,7)}
        self.cost_func_normal = CostFunction_normal(self.params)
        self.cost_func_evac = CostFunction_evacuation(self.params)
        self.background_ratio = self.params.background_ratio
        self.reset()

    def reset(self):
        self.step = self.sim_start_step
        self.x = np.zeros(2*(self.params.num_zones**2) + self.params.num_zones)
        self.x_background = np.zeros((self.params.num_zones**2 + self.params.num_zones))
        self.u = np.zeros(self.params.num_zones * (self.params.num_zones - 1))
        self.xs = []
        self.xs_background = []
        self.ttt = 0.0
        self.tet = 0.0
        self.ttt_list = []
        self.tet_list = []
        self.throughput_list = []
        self.throughput_background_list = []
        self.dx_list = []
        self.dx_background_list = []
        self.n_check = []
        self.Q = copy.deepcopy(self.params.Q)[:,:,:self.params.simulation_end_time]
        self.Q_background = self.background_ratio * copy.deepcopy(self.params.Q_background)[:,:,:self.params.simulation_end_time]
        self.non_evac_Q = np.zeros((self.params.num_zones, self.params.num_zones))
        self.not_started_background = np.zeros((self.params.num_zones, self.params.num_zones))
                
    def mfd(self, N: np.ndarray):
        return mfd(np.clip(N, 0, self.params.N_jam), self.params.A, self.params.B, self.params.C)
    
    def mfd_zone(self, N_zone: float, zone: int):
        N_zone = np.clip(N_zone, 0, self.params.N_jam[zone])
        return self.params.A[zone] * (N_zone**3) + self.params.B[zone] * (N_zone**2) + self.params.C[zone] * N_zone

    def critical_accumulation(self):
        a, b, c = self.params.A, self.params.B, self.params.C
        return (-b - np.sqrt(b**2 - 3*a*c)) / (3*a)
    
    def parking_success_func(self, N_p: np.ndarray, N_s: np.ndarray, Cap_i: np.ndarray, N_all: np.ndarray):
        return self.params.Ap * (N_s ** self.params.eta1) * ((np.maximum(Cap_i - N_p, 0)/100) ** self.params.eta2) * (np.divide(self.mfd(N_all), N_all, where= (N_all!=0)) ** self.params.eta3)
        
    def k_shortest_theta(self, N: np.ndarray):
        return optimized_k_shortest_theta(N, self.params.path_dict, self.params.L_m, self.params.num_zones, self.params.A, self.params.B, self.params.C, self.params.max_boundary_capacity)
    
    def vector_to_matrix(self, vector: np.ndarray, num_zones: int):
        mat = np.zeros((num_zones, num_zones))
        mat[np.where(~np.eye(num_zones, dtype=bool))] = vector
        return mat
    
    def matrix_to_vector(self, mat: np.ndarray, num_zones: int):
        return mat[np.where(~np.eye(num_zones, dtype=bool))]
    
    def run_simulation(self):
        self.reset()
        for _ in range(self.sim_start_step, self.sim_end_step):
            self.step_simulation()
        self.risk_people = self.cost_func_evac.risk_people(self.x, self.non_evac_Q.sum(axis=1))
    
    def step_simulation(self, noise_x: np.ndarray = None, noise_x_background: np.ndarray = None):        
        dx, dx_background = self.state_transition(self.x, self.x_background, self.u, self.step)
        dx = np.array(dx).flatten()
        dx_background = np.array(dx_background).flatten()
        np.add(self.x, self.sampling_time * dx, out=self.x)
        np.add(self.x_background, self.sampling_time * dx_background, out=self.x_background)
        if noise_x is not None:
            self.x += self.x * noise_x
        if noise_x_background is not None:
            self.x_background += self.x_background * noise_x_background
        np.clip(self.x, 0, None, out=self.x)
        np.clip(self.x_background, 0, None, out=self.x_background)
        self.xs.append(self.x.copy())
        self.xs_background.append(self.x_background.copy())
        self.tet += self.sampling_time * self.cost_func_evac.tet(self.x)
        self.ttt += self.sampling_time * self.cost_func_normal.ttt(self.x_background)
        self.step += self.sampling_time
        
        self.ttt_list.append(self.ttt)
        self.tet_list.append(self.tet)
        self.throughput_list.append(self.throughput)
        self.throughput_background_list.append(self.throughput_background)
        self.dx_list.append(dx)
        self.dx_background_list.append(dx_background)
            
    def state_transition(self, x: np.ndarray, x_background: np.ndarray, u: np.ndarray, t: int):
        num_zones = self.params.num_zones
        M_ms, M_o, M_sp, M_d, M_mp_back, M_o_back = self.compute_M(x, x_background, t)
        self.throughput = M_sp.tolist()
        self.throughput_background = M_mp_back.tolist()
        
        # Modify self.Q to prevent demand to zones where N_p exceeds Cap_i
        N_p = x[num_zones**2 + num_zones : num_zones**2 + 2 * num_zones]
        excess_zone = np.where(N_p >= self.params.Cap_i)[0]
        N_re_o, N_re_d, M_so = np.zeros((num_zones, num_zones)), np.zeros((num_zones, num_zones)), np.zeros(num_zones)
        if len(excess_zone) > 0:
            self.Q[:,excess_zone,t] = 0
            od_df = self.params.od_df[self.params.od_df['time'] == t].astype(int)
            for zone in excess_zone:
                df_zone = od_df[od_df['destination'] == zone]
                for _, row in df_zone.iterrows():
                    origin, value = row['origin'], row['value']
                    alternatives = [row['dest_alt1'], row['dest_alt2'], row['dest_alt3']]
                    valid_alts = [alt for alt in alternatives if alt != 999999 and alt not in excess_zone]
                    
                    if valid_alts != [] and valid_alts[0] != 999999:
                        self.Q[origin, valid_alts[0], t] += value  # Assign to the first valid alternative destination
                    else:
                        self.non_evac_Q[origin, zone] += value  # Assign to non-evacuation if all alternatives are invalid
            
            # Modify M_o, M_d, and N_s
            N_s = x[num_zones : 2 * num_zones]
            N_d = x[num_zones**2 + 2 * num_zones : ]
            # Randomly distribute traffic from N_s[excess_zone] to M_o
            N_re_o = redistribute_traffic(N_s, excess_zone, self.params.L_o)
            M_so[excess_zone] = N_s[excess_zone]
            N_re_d = np.zeros((num_zones, num_zones))
            if self.vector_to_matrix(N_d, num_zones)[excess_zone].sum() > 0:
                N_re_d = redistribute_traffic(N_d, excess_zone, self.params.L_o)
                
        # N_m
        sum_M_hii = M_o.sum(axis=0)
        dN_m = np.diag(self.Q[:,:,t]) + np.diag(sum_M_hii) - M_ms 
        sum_M_hii_back = M_o_back.sum(axis=0)
        dN_m_back = np.diag(self.Q_background[:,:,t]) + np.diag(sum_M_hii_back) - M_mp_back 
        
        # N_s
        dN_s = M_ms - M_sp - M_so
        
        # N_o
        dN_o = self.Q[:,:,t] + sum_M_hii - M_o.sum(axis=1) + N_re_o 
        dN_o_back = self.Q_background[:,:,t] + sum_M_hii_back - M_o_back.sum(axis=1) 
        # dN_oの対角成分は0
        dN_o = np.where(np.eye(num_zones, dtype=bool), 0, dN_o)
        dN_o_back = np.where(np.eye(num_zones, dtype=bool), 0, dN_o_back)
        
        # N_p
        omega_mat = self.vector_to_matrix(u, num_zones)
        dN_p = M_sp - self.params.compliance_rate * omega_mat.sum(axis=1) + np.diag(M_d.sum(axis=0))
        dN_p_back = M_mp_back
        
        # N_d
        dN_d = self.params.compliance_rate * omega_mat + M_d.sum(axis=0) - M_d.sum(axis=1) + N_re_d
        dN_d = np.where(np.eye(num_zones, dtype=bool), 0, dN_d)
        
        # Ensure that the sum of N_m, N_s, N_o, and N_d does not exceed N_all
        dN_all = dN_m + dN_s + dN_o.sum(axis=1) + dN_d.sum(axis=1) + dN_m_back + dN_o_back.sum(axis=1)
        new_N_all = self.N_all + dN_all
        if np.any(new_N_all > self.params.N_jam):
            # Reduce demand Q proportionally to N_all - N_jam where N_all exceeds N_jam
            excess = np.where(new_N_all - self.params.N_jam > 0, new_N_all - self.params.N_jam, 0)
            revision_Q = excess * np.divide((self.N_all_evac + dN_m + dN_s + dN_o.sum(axis=1) + dN_d.sum(axis=1)), new_N_all, where=(new_N_all > 0))
            revision_Q_background = excess - revision_Q
            assert np.any(revision_Q > 0) or np.any(revision_Q_background > 0)
            # Distribute revision_Q proportionally along axis=1 of self.Q
            new_Q, new_Q_background = copy.deepcopy(self.Q), copy.deepcopy(self.Q_background)
            denominator = self.Q[:,:,t].sum(axis=1)[:, np.newaxis]
            new_Q[:,:,t] -= np.where(
                np.tile(denominator, (1, num_zones)) != 0, 
                np.divide(self.Q[:,:,t], denominator, where=(denominator != 0)) * revision_Q[:,np.newaxis],
                0)
            denominator = self.Q_background[:,:,t].sum(axis=1)[:, np.newaxis]
            new_Q_background[:,:,t] -= np.where(
                np.tile(denominator, (1, num_zones)) != 0, 
                np.divide(self.Q_background[:,:,t], denominator, where=(denominator != 0)) * revision_Q_background[:,np.newaxis],
                0)
            new_Q = np.where(new_Q < 0, 0, new_Q)
            new_Q_background = np.where(new_Q_background < 0, 0, new_Q_background)
            if t < self.sim_end_step - 1:
                residual = self.Q[:,:,t] - new_Q[:,:,t]
                residual_background = self.Q_background[:,:,t] - new_Q_background[:,:,t]
                max_time = np.minimum(t+1+360, self.sim_end_step)
                for restart_time in range(t+1, max_time):
                    self.Q[:,:,restart_time] += residual / (max_time - t)
                    self.Q_background[:,:,restart_time] += residual_background / (max_time - t)
                self.Q[:,:,t] = new_Q[:,:,t]
                self.Q_background[:,:,t] = new_Q_background[:,:,t]
            else:
                self.non_evac_Q += self.Q[:,:,t] - new_Q[:,:,t]
                self.not_started_background += self.Q_background[:,:,t] - new_Q_background[:,:,t]
                self.Q[:,:,t] = new_Q[:,:,t]
                self.Q_background[:,:,t] = new_Q_background[:,:,t]
                
        # recalculate dN_m, dN_s, dN_o, dN_p, dN_d
        dN_m = np.diag(self.Q[:,:,t]) + np.diag(sum_M_hii) - M_ms 
        dN_m_back = np.diag(self.Q_background[:,:,t]) + np.diag(sum_M_hii_back) - M_mp_back
        dN_o = self.Q[:,:,t] + sum_M_hii - M_o.sum(axis=1) + N_re_o 
        dN_o_back = self.Q_background[:,:,t] + sum_M_hii_back - M_o_back.sum(axis=1) 
        
        dN_o = self.matrix_to_vector(dN_o, num_zones) 
        dN_o_back = self.matrix_to_vector(dN_o_back, num_zones) 
        dN_d = self.matrix_to_vector(dN_d, num_zones)
        
        return np.concatenate((dN_m, dN_s, dN_o, dN_p, dN_d), axis=None), np.concatenate((dN_m_back, dN_o_back, dN_p_back), axis=None)
    
    def compute_M(self, x: np.ndarray, x_background: np.ndarray, t: int):
        num_zones = self.params.num_zones
        N_m = x[:num_zones]
        N_s = x[num_zones : 2 * num_zones]
        N_o = x[2 * num_zones : num_zones**2 + num_zones]
        N_p = x[num_zones**2 + num_zones : num_zones**2 + 2 * num_zones]
        N_d = x[num_zones**2 + 2 * num_zones : ]
        N_m_background = x_background[:num_zones]
        N_o_background = x_background[num_zones : num_zones**2]
        
        N_o_mat = self.vector_to_matrix(N_o, num_zones) 
        N_o_background_mat = self.vector_to_matrix(N_o_background, num_zones) 
        N_d_mat = self.vector_to_matrix(N_d, num_zones) 
        self.N_all_evac = N_m + N_s + N_o_mat.sum(axis=1) + N_d_mat.sum(axis=1)
        self.N_all_normal = N_m_background + N_o_background_mat.sum(axis=1)
        self.N_all = self.N_all_evac + self.N_all_normal 
        self.n_check.append(np.round(self.N_all / self.params.N_jam, 2))
        
        # update theta_o and theta_d every path_update_interval steps
        if t % self.params.path_update_interval == 0:
            self.params.theta_o = self.k_shortest_theta(self.N_all)
            self.params.theta_d = self.params.theta_o.copy()
        
        outflow = np.clip(self.mfd(self.N_all) / self.params.L_m, 0, self.N_all)
        M_ms = np.divide(N_m, self.N_all, where=self.N_all!=0) * outflow
        if N_d.sum() != 0:
            M_o, M_o_back, M_d = self.compute_o_d_matrix(N_o, N_o_background, N_d, self.N_all)
        else:
            M_o, M_o_back = self.compute_o_matrix(N_o, N_o_background, self.N_all)
            M_d = np.zeros((num_zones, num_zones, num_zones))
        
        M_sp = np.where(N_p < self.params.Cap_i, 
                        np.minimum(self.parking_success_func(N_p, N_s, self.params.Cap_i, self.N_all), N_s),
                        0)
        
        M_mp_back = np.divide(N_m_background,self.N_all, where=(self.N_all!=0)) * outflow
        
        return M_ms, M_o, M_sp, M_d, M_mp_back, M_o_back
        
    def compute_o_matrix(self, N_o: np.ndarray, N_o_background: np.ndarray, N_all: np.ndarray):
        num_zones = self.params.num_zones
        outflow = np.tile(self.mfd(N_all)[:,np.newaxis], (1, num_zones)) / self.params.L_o
        outflow_ij = np.divide(self.vector_to_matrix(N_o, num_zones), N_all[:,np.newaxis], where=(N_all[:,np.newaxis]!=0)) * outflow
        outflow_ij = np.where(np.isnan(outflow_ij), 0, outflow_ij)
        assert np.isnan(outflow_ij).sum() == 0, f"outflow_ij: {outflow_ij}, N_o: {N_o}, N_all: {N_all}"
        M_o = self.params.theta_o * outflow_ij[:,np.newaxis,:]
        outflow_ij_background = np.divide(self.vector_to_matrix(N_o_background, num_zones), N_all[:,np.newaxis], where=(N_all[:,np.newaxis]!=0)) * outflow
        outflow_ij_background = np.where(np.isnan(outflow_ij_background), 0, outflow_ij_background)
        assert np.isnan(outflow_ij_background).sum() == 0, f"outflow_ij_background: {outflow_ij_background}, N_o_background: {N_o_background}, N_all: {N_all}"
        M_o_background = self.params.theta_o * outflow_ij_background[:,np.newaxis,:]
        for r in range(num_zones):
            M_o[r,:,r] = 0 
            M_o_background[r,:,r] = 0 
        
        # Correction of outflow traffic. Ensure outflow traffic does not exceed boundary capacity
        max_boundary_capacity = self.params.max_boundary_capacity
        boundary_capacity = np.zeros((num_zones,num_zones))
        N_jam = self.params.N_jam
        alpha = self.params.alpha
        adj_mask = self.params.adj_matrix == 1
        N_all_broadcasted = np.tile(N_all, (num_zones, 1))
        N_jam_broadcasted = np.tile(N_jam, (num_zones, 1))
        condition1 = (0 <=  N_all_broadcasted) & (N_all_broadcasted < alpha * N_jam_broadcasted) & adj_mask
        condition2 = (alpha * N_jam_broadcasted <= N_all_broadcasted) & (N_all_broadcasted <= N_jam_broadcasted) & adj_mask
        boundary_capacity[condition1] = max_boundary_capacity[condition1]
        boundary_capacity[condition2] = max_boundary_capacity[condition2] * (1/(1 - alpha)) * (1 - (N_all_broadcasted[condition2] / N_jam_broadcasted[condition2]))
        boundary_capacity = np.where(boundary_capacity < 1e-05, 0, boundary_capacity)
        M_o_ih = M_o.sum(axis=2)
        M_o_background_ih = M_o_background.sum(axis=2)
        M_o_ih_mod = np.minimum(M_o_ih + M_o_background_ih, boundary_capacity)
        
        ratio_mat_all = np.divide(M_o_ih_mod, M_o_ih + M_o_background_ih, where=(M_o_ih + M_o_background_ih != 0)) # Total correction amount
        ratio_mat = np.divide(ratio_mat_all * M_o_ih, M_o_ih + M_o_background_ih, where=(M_o_ih + M_o_background_ih != 0)) # correction amount of M_o
        ratio_mat_background = np.divide(ratio_mat_all * M_o_background_ih, M_o_ih + M_o_background_ih, where=(M_o_ih + M_o_background_ih != 0)) # correction amount of M_o_background
        ratio_mat = np.where((ratio_mat < 1e-05) | np.isnan(ratio_mat), 0, ratio_mat) 
        ratio_mat_background = np.where((ratio_mat_background < 1e-05) | np.isnan(ratio_mat_background), 0, ratio_mat_background) 
        M_o *= ratio_mat[:, :, np.newaxis]
        M_o_background *= ratio_mat_background[:, :, np.newaxis]
        
        return M_o, M_o_background
    
    def compute_o_d_matrix(self, N_o: np.ndarray, N_o_background: np.ndarray, N_d: np.ndarray, N_all: np.ndarray):
        num_zones = self.params.num_zones
        outflow = np.tile(self.mfd(N_all)[:,np.newaxis], (1, num_zones)) / self.params.L_o
        outflow_ij = np.divide(self.vector_to_matrix(N_o, num_zones), N_all[:,np.newaxis], where=(N_all[:,np.newaxis]!=0)) * outflow
        M_o = self.params.theta_o * outflow_ij[:,np.newaxis,:]
        outflow_ij_background = np.divide(self.vector_to_matrix(N_o_background, num_zones), N_all[:,np.newaxis], where=(N_all[:,np.newaxis]!=0)) * outflow
        M_o_background = self.params.theta_o * outflow_ij_background[:,np.newaxis,:]
        outflow_ij_d = np.divide(self.vector_to_matrix(N_d, num_zones), N_all[:,np.newaxis], where=(N_all[:,np.newaxis]!=0)) * outflow
        M_d = self.params.theta_d * outflow_ij_d[:,np.newaxis,:]
        for r in range(num_zones):
            M_o[r,:,r] = 0 
            M_o_background[r,:,r] = 0 
            M_d[r,:,r] = 0
        
        # Correction of outflow traffic. Ensure outflow traffic does not exceed boundary capacity
        max_boundary_capacity = self.params.max_boundary_capacity
        boundary_capacity = np.zeros((num_zones,num_zones))
        N_jam = self.params.N_jam
        alpha = self.params.alpha
        adj_mask = self.params.adj_matrix == 1
        N_all_broadcasted = np.tile(N_all, (num_zones, 1))
        N_jam_broadcasted = np.tile(N_jam, (num_zones, 1))
        condition1 = (0 <=  N_all_broadcasted) & (N_all_broadcasted < alpha * N_jam_broadcasted) & adj_mask
        condition2 = (alpha * N_jam_broadcasted <= N_all_broadcasted) & (N_all_broadcasted <= N_jam_broadcasted) & adj_mask
        boundary_capacity[condition1] = max_boundary_capacity[condition1]
        boundary_capacity[condition2] = max_boundary_capacity[condition2] * (1/(1 - alpha)) * (1 - (N_all_broadcasted[condition2] / N_jam_broadcasted[condition2]))
        boundary_capacity = np.where(boundary_capacity < 1e-05, 0, boundary_capacity) 
        M_o_ih = M_o.sum(axis=2)
        M_o_background_ih = M_o_background.sum(axis=2)
        M_d_ih = M_d.sum(axis=2)
        M_o_ih_mod = np.minimum(M_o_ih + M_o_background_ih + M_d_ih, boundary_capacity) 
        ratio_mat_all = np.divide(M_o_ih_mod, M_o_ih + M_o_background_ih + M_d_ih, where=(M_o_ih + M_o_background_ih + M_d_ih != 0)) 
        ratio_mat = np.divide(ratio_mat_all * M_o_ih, M_o_ih + M_o_background_ih + M_d_ih, where=(M_o_ih + M_o_background_ih + M_d_ih != 0)) 
        ratio_mat_background = np.divide(ratio_mat_all * M_o_background_ih, M_o_ih + M_o_background_ih + M_d_ih, where=(M_o_ih + M_o_background_ih + M_d_ih != 0))
        ratio_mat_d = np.divide(ratio_mat_all * M_d_ih, M_o_ih + M_o_background_ih + M_d_ih, where=(M_o_ih + M_o_background_ih + M_d_ih != 0)) 
        ratio_mat = np.where((ratio_mat < 1e-05) | np.isnan(ratio_mat), 0, ratio_mat) 
        ratio_mat_background = np.where((ratio_mat_background < 1e-05) | np.isnan(ratio_mat_background), 0, ratio_mat_background) 
        ratio_mat_d = np.where((ratio_mat_d < 1e-05) | np.isnan(ratio_mat_d), 0, ratio_mat_d) 
        M_o *= ratio_mat[:, :, np.newaxis]
        M_o_background *= ratio_mat_background[:, :, np.newaxis]
        M_d *= ratio_mat_d[:, :, np.newaxis]
        
        return M_o, M_o_background, M_d
    
    def flow_visualize(self):
        if self.xs == [] and self.xs_background == []:
            self.run_simulation()
        num_zones = self.params.num_zones
        graph = nx.DiGraph()
        graph.add_nodes_from([i for i in range(num_zones)])
        graph.add_edges_from([(i, j) for i in range(num_zones) for j in range(num_zones) if self.params.adj_matrix[i, j] == 1])
        
        edge_labels = {}
        for i, j in graph.edges():
            edge_labels[(i, j)] = 0
        t = self.sim_start_step
        for x, x_background in zip(self.xs, self.xs_background):
            _, M_o, _, _, _, M_o_back = self.compute_M(x, x_background, t)
            for i, j in graph.edges():
                edge_labels[(i, j)] += M_o[i, j, :].sum() + M_o_back[i, j, :].sum()
            t += self.sampling_time
        
        # visualize link flow with networkx graph
        plt.figure(figsize=(10, 7))
        nx.draw_networkx_nodes(graph, self.pos, node_color="lightblue", node_size=300)
        nx.draw_networkx_edges(graph, self.pos, edge_color="black",
                               arrowstyle="->",connectionstyle="arc3,rad=0.1",
                               width=[edge_labels[(i, j)]/max(edge_labels.values())*7 for i, j in graph.edges()])
        nx.draw_networkx_labels(graph, self.pos)
        plt.title("Flow")
        if self.output_path is not None:
            plt.savefig(f'{self.output_path}/flow.png')
            plt.close()
        else:
            plt.show()
    
    def capacity_visualize(self):
        num_zones = self.params.num_zones
        graph = nx.DiGraph()
        graph.add_nodes_from([i for i in range(num_zones)])
        graph.add_edges_from([(i, j) for i in range(num_zones) for j in range(num_zones) if self.params.adj_matrix[i, j] == 1])
        
        edge_labels = {}
        for i, j in graph.edges():
            edge_labels[(i, j)] = self.params.max_boundary_capacity[i, j]
        
        plt.figure(figsize=(10, 7))
        nx.draw_networkx_nodes(graph, self.pos, node_color="lightblue", node_size=300)
        nx.draw_networkx_edges(graph, self.pos, edge_color="black",
                               arrowstyle="->",connectionstyle="arc3,rad=0.1",
                               width=[edge_labels[(i, j)]/max(edge_labels.values())*7 for i, j in graph.edges()])
        nx.draw_networkx_labels(graph, self.pos)
        plt.title("Max Boundary Capacity")
        if self.output_path is not None:
            plt.savefig(f'{self.output_path}/max_capacity.png')
            plt.close()
        else:
            plt.show()
    
    
    def plot_accumulation(self):
        if self.output_path is not None:
            os.makedirs(f"{self.output_path}/accumulation", exist_ok=True)
        num_zones = self.params.num_zones
        N_m = [x[:num_zones] for x in self.xs]
        N_s = [x[num_zones : 2 * num_zones] for x in self.xs]
        N_o = [x[2 * num_zones : num_zones**2 + num_zones] for x in self.xs]
        N_p = [x[num_zones**2 + num_zones : num_zones**2 + 2 * num_zones] for x in self.xs]
        N_d = [x[num_zones**2 + 2 * num_zones : ] for x in self.xs]       
        N_m_background = [x[:num_zones] for x in self.xs_background]
        N_o_background = [x[num_zones : num_zones**2] for x in self.xs_background]
        N_p_background = [x[num_zones**2 : ] for x in self.xs_background]
        N_o_i = [[sum(N_o[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(len(self.xs))]
        N_d_i = [[sum(N_d[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(len(self.xs))]
        N_o_background_i = [[sum(N_o_background[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(len(self.xs))]
        
        # Plot
        tgrid = [k/60 for k in range(self.sim_start_step, self.sim_end_step, self.sampling_time)]
        
        fig, ax = plt.subplots(figsize=(10, 7))
        for i in range(num_zones):
            n_check = [n[i] for n in self.n_check]
            ax.plot(tgrid, n_check, '-', color=self.params.color_dict[i], label=f'zone {i}')
        ax.hlines(1, 0, tgrid[-1], "black", linestyles='dashed')
        ax.set_xticks(np.arange(0, 50, 5))
        ax.set_xlabel('t')
        ax.set_ylabel('N_all/N_jam')
        ax.legend()
        ax.grid(False)
        plt.tight_layout()
        if self.output_path is not None:
            plt.savefig(f'{self.output_path}/accumulation/Nall_Njam.png')
            plt.close()
        else:
            plt.show()

        y_limits = [self.get_y_limits(N_m),
                    self.get_y_limits(N_s),
                    self.get_y_limits(N_o),
                    self.get_y_limits(N_p),
                    self.get_y_limits(N_d),
                    self.get_y_limits(N_m_background),
                    self.get_y_limits(N_o_background),
                    self.get_y_limits(N_p_background)]

        titles = ['N_m', 'N_s', 'N_o', 'N_p', 'N_d', 'N_m_background', 'N_o_background', 'N_p_background']
        titles_tex = [r'$N^m$', r'$N^s$', r'$N^o$', r'$N^p$', r'$N^d$', r'$N^{m, back}$', r'$N^{o, back}$', r'$N^{p, back}$']
        data = [N_m, N_s, N_o, N_p, N_d, N_m_background, N_o_background, N_p_background]
        link_legend = []
        def index_to_position(n, index):
            count = 0
            for i in range(n):
                for j in range(n):
                    if i != j:  
                        if count == index:
                            return i, j
                        count += 1
            return None  

        link_flow = []
        link_flow_background = []
        for link in range(len(N_o[0])):
            link_data = [N_o[j][link] for j in range(len(N_o))]
            link_flow.append(sum(link_data))
            link_data = [N_o_background[j][link] for j in range(len(N_o_background))]
            link_flow_background.append(sum(link_data))
        link_flow = sorted(link_flow, reverse=True)
        link_flow_background = sorted(link_flow_background, reverse=True)

        for i, (d, (min_y, max_y)) in enumerate(zip(data, y_limits)):
            fig, ax = plt.subplots(figsize=(10, 7))        
            if titles[i] in ['N_o', 'N_d', 'N_o_background']:
                for link in range(len(d[0])):
                    link_data = [d[j][link] for j in range(len(d))]
                    if sum(link_data) > link_flow[10]:
                        ax.plot(tgrid, link_data, '-')
                        row, col = index_to_position(num_zones, link)
                        link_legend.append(f'{row} - {col}')
            elif titles[i]=='N_p':
                for zone_idx in range(num_zones):
                    zone_data = [d[j][zone_idx] for j in range(len(d))]
                    ax.plot(tgrid, zone_data, '-', color=self.params.color_dict[zone_idx], label=f'Zone {zone_idx}')
                for zone_idx in range(num_zones):
                    ax.hlines(self.params.Cap_i[zone_idx], 0, tgrid[-1], color=self.params.color_dict[zone_idx], linestyles='dashed', label=f'Zone {zone_idx} capacity')
            else:
                for zone_idx in range(num_zones):
                    zone_data = [d[j][zone_idx] for j in range(len(d))]
                    ax.plot(tgrid, zone_data, '-', color=self.params.color_dict[zone_idx])
            ax.set_xticks(np.arange(0, 50, 5))
            ax.set_xlabel('Time (hour)', fontsize=14)
            ax.set_ylim(min_y, max_y)
            
            if titles[i] in ['N_o', 'N_d', 'N_o_background']:
                ax.legend(link_legend, ncol=4, fontsize=14)
            elif titles[i]=='N_p':
                ax.legend(ncol=2, fontsize=14)
            else:
                legend = ['Zone ' + str(j) for j in range(num_zones)]
                ax.legend(legend, fontsize=14)
            ax.set_title(titles_tex[i], fontsize=16)
            ax.grid(False)
            plt.tight_layout()
            if self.output_path is not None:
                plt.savefig(f'{self.output_path}/accumulation/{titles[i]}.png')
                plt.close()
            else:
                plt.show()
            
        for zone in range(num_zones):
            # stacked area chart of N_all
            plt.figure(figsize=(10, 7))
            m = [N_m[t][zone] for t in range(len(self.xs))]
            s = [N_s[t][zone] for t in range(len(self.xs))]
            o = [N_o_i[t][zone] for t in range(len(self.xs))]
            d = [N_d_i[t][zone] for t in range(len(self.xs))]
            m_background = [N_m_background[t][zone] for t in range(len(self.xs))]
            o_background = [N_o_background_i[t][zone] for t in range(len(self.xs))]
            plt.stackplot(tgrid, m, s, o, d, m_background, o_background, labels=[r'$N^m$', r'$N^s$', r'$N^o$', r'$N^d$', r'$N^{m, back}$', r'$N^{o, back}$'])
            plt.xticks(np.arange(0, 50, 5))
            plt.legend(loc='upper left')
            plt.xlabel('Time (hour)', fontsize=14)
            plt.ylabel('Number of vehicles', fontsize=14)
            plt.title('Zone '+str(zone), fontsize=16)
            if self.output_path is not None:
                plt.savefig(f'{self.output_path}/accumulation/stacked_zone_{zone}.png')
                plt.close()
            else:
                plt.show()

    def plot_throughput(self):
        num_zones = self.params.num_zones
        tgrid = [k/60 for k in range(self.sim_start_step, self.sim_end_step, self.sampling_time)]
        fig, ax = plt.subplots(figsize=(10, 7))
        for i in range(num_zones):
            throughput = [x[i] for x in self.throughput_list]
            ax.plot(tgrid, throughput, '-', color=self.params.color_dict[i], label=f'zone {i}')
        ax.set_xticks(np.arange(0, 50, 5))
        ax.set_xlabel('Elapsed Time (hour)', fontsize=14)
        ax.set_ylabel('Throughput (veh/min)', fontsize=14)
        ax.legend(fontsize=14)
        ax.grid(False)
        plt.title("Throughput of evacuation vehicles", fontsize=16)
        plt.tight_layout()
        if self.output_path is not None:
            plt.savefig(f'{self.output_path}/throughput_evac.png')
            plt.close()
        else:
            plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 7))
        for i in range(num_zones):
            throughput = [x[i] for x in self.throughput_background_list]
            ax.plot(tgrid, throughput, '-', color=self.params.color_dict[i], label=f'zone {i}')
        ax.set_xticks(np.arange(0, 50, 5))
        ax.set_xlabel('Time (hour)', fontsize=14)
        ax.set_ylabel('Throughput (veh/min)', fontsize=14)
        ax.legend(fontsize=14)
        ax.grid(False)
        plt.title("Throughput of normal vehicles", fontsize=16)
        plt.tight_layout()
        if self.output_path is not None:
            plt.savefig(f'{self.output_path}/throughput_normal.png')
            plt.close()
        else:
            plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 7))
        for i in range(num_zones):
            throughput = [x[i]+xback[i] for x, xback in zip(self.throughput_list, self.throughput_background_list)]
            ax.plot(tgrid, throughput, '-', color=self.params.color_dict[i], label=f'zone {i}')
        ax.set_xticks(np.arange(0, 50, 5))
        ax.set_xlabel('Time (hour)', fontsize=14)
        ax.set_ylabel('Throughput (veh/min)', fontsize=14)
        ax.legend(fontsize=14)
        ax.grid(False)
        plt.title("Throughput of all vehicles", fontsize=16)
        plt.tight_layout()
        if self.output_path is not None:
            plt.savefig(f'{self.output_path}/throughput_all.png')
            plt.close()
        else:
            plt.show()
    
    def plot_mfd(self):
        if self.output_path is not None:
            os.makedirs(f"{self.output_path}/mfd", exist_ok=True)
        num_zones = self.params.num_zones
        N_m = [x[:num_zones] for x in self.xs]
        N_s = [x[num_zones : 2 * num_zones] for x in self.xs]
        N_o = [x[2 * num_zones : num_zones**2 + num_zones] for x in self.xs]
        N_p = [x[num_zones**2 + num_zones : num_zones**2 + 2 * num_zones] for x in self.xs]
        N_d = [x[num_zones**2 + 2 * num_zones : ] for x in self.xs]       
        N_m_background = [x[:num_zones] for x in self.xs_background]
        N_o_background = [x[num_zones : num_zones**2] for x in self.xs_background]
        N_p_background = [x[num_zones**2 : ] for x in self.xs_background]
        
        # Plot
        tgrid = [k/60 for k in range(self.sim_start_step, self.sim_end_step, self.sampling_time)]
        
        #mfd
        N_o_i = [[sum(N_o[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(len(self.xs))]
        N_d_i = [[sum(N_d[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(len(self.xs))]
        N_o_background_i = [[sum(N_o_background[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(len(self.xs))]
        N_all = [N_m[t] + N_s[t] + N_o_i[t] + N_d_i[t] + N_m_background[t] + N_o_background_i[t] for t in range(len(self.xs))]

        marker_size = 15
        alpha = 1.0
        critical_accumulation = self.critical_accumulation()
        
        for zone in range(num_zones):
            plt.figure(figsize=(10, 7))
            plt.clf()
            x = np.linspace(0, self.params.N_jam[zone], 100)
            y = self.mfd_zone(x, zone)
            plt.xlim(0, self.params.N_jam[zone])
            plt.ylim(0, max(y) + 20)
            plt.plot(x, y, color='black', linewidth=1)
            x = np.array([N_all[t][zone] for t in range(len(self.xs))])
            y = self.mfd_zone(x, zone)
            color = np.arange(self.sim_start_step//60, self.sim_end_step//60, self.sampling_time/60)
            plt.scatter(x, y, s=marker_size, c=color, cmap='viridis', alpha=alpha)
            cbar = plt.colorbar(shrink=0.8, pad=0.02)
            cbar.set_label('Time (hour)', fontsize=12)
            
            n_critical = critical_accumulation[zone]
            N_all_zone = [N_all[t][zone] for t in range(len(self.xs))]
            if np.any(N_all_zone >= n_critical):
                t_c = np.argmax(N_all_zone >= n_critical)
                t_c_all = (np.argmax(N_all_zone >= n_critical)  + self.sim_start_step )* self.sampling_time
                plt.text(N_all[t_c][zone], self.mfd_zone(N_all[t_c][zone], zone) + 10, fr'$t_{{cri}}={t_c_all/60:.1f}$', fontsize=12)
            else:
                t_max = np.argmax(N_all_zone)
                t_max_all = (np.argmax(N_all_zone) + self.sim_start_step) * self.sampling_time
                plt.text(N_all[t_max][zone], self.mfd_zone(N_all[t_max][zone], zone) + 10, fr'$t_{{max}}={t_max_all/60:.1f}$', fontsize=12)
            
            if np.any(N_all_zone >= self.params.N_jam[zone]):
                t_jam = np.argmax(N_all_zone >= self.params.N_jam[zone])
                t_jam_all = (np.argmax(N_all_zone >= self.params.N_jam[zone]) + self.sim_start_step) * self.sampling_time
                plt.text(N_all[t_jam][zone]-500,  5, fr'$t_{{jam}}={t_jam_all/60:.1f}$', fontsize=12)
            plt.xlabel('Accumulation (veh)', fontsize=14)
            plt.ylabel(fr'Production (veh $\cdot$ km / min)', fontsize=14)
            plt.title('Zone '+str(zone), fontsize=16)
            if self.output_path is not None:
                plt.savefig(f'{self.output_path}/mfd/zone_{zone}.png')
                plt.close()
            else:
                plt.show()
        
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
        axes = axes.flatten()

        for zone in range(num_zones):
            ax = axes[zone]
            x_mfd = np.linspace(0, self.params.N_jam[zone], 100)
            y_mfd = self.mfd_zone(x_mfd, zone)
            ax.plot(x_mfd, y_mfd, color='black', linewidth=1)

            x_data = np.array([N_all[t][zone] for t in range(len(self.xs))])
            y_data = self.mfd_zone(x_data, zone)
            color = np.arange(self.sim_start_step//60, self.sim_end_step//60, self.sampling_time/60)
            ax.scatter(x_data, y_data, s=marker_size, c=color, cmap='viridis', alpha=alpha)
            
            n_critical = critical_accumulation[zone]
            N_all_zone = [N_all[t][zone] for t in range(len(self.xs))]
            if np.any(N_all_zone >= n_critical):
                t_c = np.argmax(N_all_zone >= n_critical)
                t_c_all = (np.argmax(N_all_zone >= n_critical)  + self.sim_start_step )* self.sampling_time
                ax.text(N_all[t_c][zone], self.mfd_zone(N_all[t_c][zone], zone) + 10, fr'$t_{{cri}}={t_c_all/60:.1f}$', fontsize=12)
            else:
                t_max = np.argmax(N_all_zone)
                t_max_all = (np.argmax(N_all_zone) + self.sim_start_step) * self.sampling_time
                ax.text(N_all[t_max][zone], self.mfd_zone(N_all[t_max][zone], zone) + 10, fr'$t_{{max}}={t_max_all/60:.1f}$', fontsize=12)
            
            if np.any(N_all_zone >= self.params.N_jam[zone]):
                t_jam = np.argmax(N_all_zone >= self.params.N_jam[zone])
                t_jam_all = (np.argmax(N_all_zone >= self.params.N_jam[zone]) + self.sim_start_step) * self.sampling_time
                ax.text(N_all[t_jam][zone]-500,  5, fr'$t_{{jam}}={t_jam_all/60:.1f}$', fontsize=12)
            
            ax.set_xlim(0, self.params.N_jam[zone])
            ax.set_ylim(0, max(y_mfd) + 20)
            ax.set_title(f"Zone {zone}", fontsize=16)
            ax.grid(False)
        
        plt.tight_layout()
        if self.output_path is not None:
            plt.savefig(f'{self.output_path}/mfd_all.png')
            plt.close()
        else:
            plt.show()

    
    def plot_mfd_animation(self):
        if self.output_path is not None:
            os.makedirs(f"{self.output_path}/mfd", exist_ok=True)
        else:
            raise ValueError("output_path is not specified.")
        if self.output_path is not None:
            os.makedirs(f"{self.output_path}/mfd", exist_ok=True)
            os.makedirs(f"{self.output_path}/mfd/mfd_frames", exist_ok=True)
        num_zones = self.params.num_zones
        frame_interval = 30
        frame_list = list(range(0, len(self.xs), int(frame_interval//self.sampling_time)))
        frame_time = list(range(self.sim_start_step, self.sim_end_step, frame_interval))
        num_frames = len(frame_list)
        
        N_m = np.array([x[:num_zones] for x in [self.xs[i] for i in frame_list]])
        N_s = np.array([x[num_zones : 2 * num_zones] for x in [self.xs[i] for i in frame_list]])
        N_o = np.array([x[2 * num_zones : num_zones**2 + num_zones] for x in [self.xs[i] for i in frame_list]])
        N_d = np.array([x[num_zones**2 + 2 * num_zones : ] for x in [self.xs[i] for i in frame_list]])
        N_m_background = np.array([x[:num_zones] for x in [self.xs_background[i] for i in frame_list]])
        N_o_background = np.array([x[num_zones : num_zones**2] for x in [self.xs_background[i] for i in frame_list]])

        N_o_i = np.array([[np.sum(N_o[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(num_frames)])
        N_d_i = np.array([[np.sum(N_d[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(num_frames)])
        N_o_background_i = np.array([[np.sum(N_o_background[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(num_frames)])

        N_all_t = N_m + N_s + N_o_i + N_d_i + N_m_background + N_o_background_i

        x_mfd_list = [np.linspace(0, self.params.N_jam[zone], 100) for zone in range(num_zones)]
        y_mfd_list = [self.mfd_zone(x_mfd_list[zone], zone) for zone in range(num_zones)]
        
        def what_time(t):
            """
            Return the time in the format of HH:MM
            """
            if t < 24 * 60:
                return f"Day 1 {int(t // 60):02d}:{int(t % 60):02d}"
            else:
                return f"Day 2 {int(t // 60 - 24):02d}:{int(t % 60):02d}"
            
        time_list = [what_time(t) for t in frame_time]
        
        for idx, t in enumerate(tqdm(frame_time, total=num_frames, desc="Creating PNG frames")):
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
            axes = axes.flatten()

            for zone in range(num_zones):
                ax = axes[zone]
                ax.plot(x_mfd_list[zone], y_mfd_list[zone], color='black', linewidth=1)

                x_data = N_all_t[:idx+1, zone]
                y_data = self.mfd_zone(x_data, zone)
                color = np.arange(idx+1)
                ax.scatter(x_data, y_data, s=15, c=color, cmap='viridis', alpha=1)
                ax.set_xlim(0, self.params.N_jam[zone])
                ax.set_ylim(0, max(y_mfd_list[zone]) + 20)
                ax.set_title(f"Zone {zone}")
                ax.grid(False)

            fig.suptitle(fr'{t/60:.1f} hour ({time_list[idx]})', fontsize=14)
            plt.tight_layout()

            frame_filename = os.path.join(f"{self.output_path}/mfd/mfd_frames", f"{t}.png")
            plt.savefig(frame_filename)
            plt.close(fig)

        print("Creating GIF...")
        frame_files = sorted([f for f in os.listdir(f"{self.output_path}/mfd/mfd_frames") if f.endswith(".png")], key=lambda x: int(x.split(".")[0]))
        images = [imageio.v2.imread(os.path.join(self.output_path, "mfd/mfd_frames", filename)) for filename in tqdm(frame_files, desc="Processing GIF")]
        imageio.mimsave(os.path.join(self.output_path, "mfd_plot.gif"), images, fps=10)

        
    def plot_time(self):
        tgrid = [k/60 for k in range(self.sim_start_step, self.sim_end_step, self.sampling_time)]
        plt.figure(figsize=(10, 7))
        cumsum = self.Q_background[:,:,self.sim_start_step:self.sim_end_step].sum(axis=(0,1)).cumsum()
        plt.plot(tgrid, np.divide(self.ttt_list, cumsum, where=cumsum!=0), label='Normal traffic')
        cumsum = self.Q[:,:,self.sim_start_step:self.sim_end_step].sum(axis=(0,1)).cumsum()
        plt.plot(tgrid, np.divide(self.tet_list, cumsum, where=cumsum!=0), label='Evacuation traffic')
        plt.xlabel('Elapsed time (hour)', fontsize=14)
        plt.ylabel('Average travel time (min)', fontsize=14)
        plt.legend(fontsize=14)
        plt.title('Average travel time of normal/evacuation vehicles', fontsize=16)
        if self.output_path is not None:
            plt.savefig(f'{self.output_path}/time.png')
            plt.close()
        else:
            plt.show()
        
        
    def get_y_limits(self, data, margin=50):
        min_val = min(min(d) for d in data) - margin
        max_val = max(max(d) for d in data) + margin
        return min_val, max_val
    
    def plot_jam(self, polygon_path):
        if self.output_path is not None:
            os.makedirs(f"{self.output_path}/jam", exist_ok=True)
        else:
            print("output_path is None")
            return
        gdf = gpd.read_file(polygon_path)
        frame_interval = 30
        frame_list = list(range(0, len(self.xs), int(frame_interval//self.sampling_time)))
        frame_time = list(range(self.sim_start_step, self.sim_end_step, frame_interval))
        num_frames = len(frame_list)
        num_zones = self.params.num_zones
        
        N_m = np.array([x[:num_zones] for x in [self.xs[i] for i in frame_list]])
        N_s = np.array([x[num_zones : 2 * num_zones] for x in [self.xs[i] for i in frame_list]])
        N_o = np.array([x[2 * num_zones : num_zones**2 + num_zones] for x in [self.xs[i] for i in frame_list]])
        N_d = np.array([x[num_zones**2 + 2 * num_zones : ] for x in [self.xs[i] for i in frame_list]])
        N_m_background = np.array([x[:num_zones] for x in [self.xs_background[i] for i in frame_list]])
        N_o_background = np.array([x[num_zones : num_zones**2] for x in [self.xs_background[i] for i in frame_list]])

        N_o_i = np.array([[np.sum(N_o[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(num_frames)])
        N_d_i = np.array([[np.sum(N_d[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(num_frames)])
        N_o_background_i = np.array([[np.sum(N_o_background[t][i * (num_zones-1) : i * (num_zones-1) + (num_zones-1)]) for i in range(num_zones)] for t in range(num_frames)])

        N_all_t = N_m + N_s + N_o_i + N_d_i + N_m_background + N_o_background_i
        
        jam_df = pd.DataFrame(
            data=N_all_t.T / self.params.N_jam.reshape(-1, 1), 
            columns=[f'jam_{t}' for t in frame_time]
        )
        gdf = pd.concat([gdf, jam_df], axis=1)
        
        def what_time(t):
            """
            Return the time in the format of HH:MM
            """
            if t < 24 * 60:
                return f"Day 1 {int(t // 60):02d}:{int(t % 60):02d}"
            else:
                return f"Day 2 {int(t // 60 - 24):02d}:{int(t % 60):02d}"
            
        time_list = [what_time(t) for t in frame_time]
        
        def jam_animation(gdf, output_path):
            for idx, t in enumerate(tqdm(frame_time, desc="Creating PNG frames")):
                fig, ax = plt.subplots(figsize=(8, 8))
                gdf.plot(ax=ax, column=f'jam_{t}', cmap='Reds', linewidth=0.5, edgecolor="black", alpha=1.0, vmin=0, vmax=1) 
                sm = cm.ScalarMappable(cmap=plt.cm.Reds, norm=mcolors.Normalize(vmin=0, vmax=1))
                cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
                cbar.set_label(fr"Accumulation ratio ($N_{{i}}(t) / N_{{i}}^{{jam}}$)", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                
                ax.set_title(fr'{t/60:.1f} hour ({time_list[idx]})')
                plt.savefig(f'{output_path}/{t}.png')
                plt.close()
        
        jam_animation(gdf, f"{self.output_path}/jam")

        print("Creating GIF...")
        images = []
        frame_files = sorted([f for f in os.listdir(f"{self.output_path}/jam") if f.endswith(".png")], key=lambda x: int(x.split(".")[0]))

        images = [imageio.v2.imread(os.path.join(self.output_path, "jam", filename)) for filename in tqdm(frame_files, desc="Processing GIF")]
        imageio.mimsave(os.path.join(self.output_path, "jam_plot.gif"), images, fps=10)

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    params = Parameters()
    output_path = '../../output/mfd_dynamics/test'
    contraflow = True
    sim = MFD_Dynamics(params, output_path=output_path)
    
    if contraflow:
        link_indices = [(i, j) for i in range(sim.params.num_zones) for j in range(sim.params.num_zones) if sim.params.adj_matrix[i, j] == 1]
        individual = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        for idx, (i, j) in enumerate(link_indices):
            if individual[idx] == 1:
                sim.params.max_boundary_capacity[i, j] *= params.contra_ratio
                sim.params.max_boundary_capacity[j, i] *= 2 - params.contra_ratio
    
    start = time.time()
    sim.run_simulation()
    print(f"elapsed_time: {time.time() - start}")
    print(f"average evacuation time: {sim.tet / sim.Q[:,:,sim.sim_start_step:sim.sim_end_step].sum()}")
    print(f"people in risk areas: {sim.risk_people}")
    print(f"average normal time: {sim.ttt / sim.Q_background[:,:,sim.sim_start_step:sim.sim_end_step].sum()}")
    print(f"people not started: {sim.not_started_background.sum()}")
    if output_path != None:
        with open(f'{output_path}/result.txt', 'w') as f:
            f.write(f"elapsed_time: {time.time() - start}\n")
            f.write(f"average evacuation time: {sim.tet / sim.Q[:,:,sim.sim_start_step:sim.sim_end_step].sum()}\n")
            f.write(f"people in risk areas: {sim.risk_people}\n")
            f.write(f"average normal time: {sim.ttt / sim.Q_background[:,:,sim.sim_start_step:sim.sim_end_step].sum()}\n")
            f.write(f"people not started: {sim.not_started_background.sum()}\n")
    sim.plot_accumulation()
    sim.plot_mfd()
    sim.plot_mfd_animation()
    sim.plot_time()
    sim.plot_throughput()
    sim.plot_jam("../../data/processed/zone_polygon.geojson")
