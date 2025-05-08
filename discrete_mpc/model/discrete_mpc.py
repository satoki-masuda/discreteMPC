import copy
import time
import gc
import ray
import os
import numpy as np
import pandas as pd
from graphillion import GraphSet
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from mfd_dynamics import MFD_Dynamics
from parameters_ndp import Parameters
from reconf_horizon import ReconfigZDD_MPC

class CrossEntropy():
    """
        Find the optimal transition pattern of discrete control variables (0/1) for each link*time step using Cross Entropy Method (CEM).

        Args:
            params: Instance of Parameters class
            initial_graph: Initial graph
            time_steps: Number of time steps in control horizon 
            seq_interval: Time interval for control sequence
            control_start_time: Start time of control sequence
            control_end_time: End time of control sequence
            prediction_end_time: End time of prediction horizon
            noise_scale: Scale of noise for state variables
            output_path: Path to save output files
        Methods:
            prepare_noise: Prepare noise for state variables
            make_graph: Create a graph from the link indices
            constraints_ZDD: Generate ZDD for transition constraints
            array_to_zdd: Convert 1D or 2D array to ZDD
            zdd_to_array: Convert ZDD to 1D or 2D array
            compute_cost: Compute cost from simulation results
            const_edge_idx: Get indices of constant edges
            evaluate_candidate: Evaluate a candidate solution
            plot_p: Plot the probability distribution of control variables
            fully_adaptive_cross_entropy: Perform CEM to find the optimal control sequence
            sim_mpc_step: Simulate the system with the best control sequence
            sim_no_policy: Simulate the system without any control policy
            sim_best_policy: Simulate the system with the best control policy
        Returns:
            best_control_sequence: Array of optimal control variables (shape: [time_steps, num_links])
            best_cost: Corresponding cost
    """
    def __init__(self, params, initial_graph, time_steps, seq_interval, control_start_time, control_end_time, prediction_end_time, noise_scale, output_path='../../output/mpc/mpc'):
        self.params = params
        self.alpha = 0.7
        self.sim = MFD_Dynamics(params, output_path=None)
        self.initial_graph = initial_graph
        self.time_steps = time_steps
        self.seq_interval = seq_interval
        self.original_capacity = copy.deepcopy(self.params.max_boundary_capacity)
        self.control_start_time = control_start_time
        self.control_end_time = control_end_time
        self.prediction_end_time = prediction_end_time
        self.p = None
        self.noise_scale = noise_scale
        self.prepare_noise()
        
        num_zones = self.params.num_zones
        self.link_indices = [(i, j) for i in range(num_zones) for j in range(num_zones) if self.params.adj_matrix[i, j] == 1]
        self.num_links = len(self.link_indices)
        self.initial_link_indices = [(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if initial_graph[idx] == 1]
        self.make_graph()
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def prepare_noise(self):
        np.random.seed(42)
        steps = self.sim.sim_end_step - self.sim.sim_start_step
        self.noise_x = self.noise_scale * np.random.normal(0, 1, size=(steps, *self.sim.x.shape))
        self.noise_x_background = self.noise_scale * np.random.normal(0, 1, size=(steps, *self.sim.x_background.shape))
    
    def make_graph(self):
        self.G = nx.Graph()
        self.G_edges = set()
        for u, v in self.link_indices:
            self.G.add_edge(f"{u}_out", f"{v}_in")
            self.G_edges.add((f"{u}_out", f"{v}_in"))
            
        GraphSet.set_universe(self.G_edges)

    def constraints_ZDD(self, constraints_csv=None):
        self.reconf = ReconfigZDD_MPC(self.params, self.initial_graph, k=self.time_steps, constraints_csv=constraints_csv, output_path=self.output_path+"/reconf")
        return self.reconf.reconfiguration()

    def array_to_zdd(self, array):
        if len(array.shape) == 1:
            assert array.shape[0] == self.num_links
            gs = [[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if array[idx] == 1]]
        else:
            assert array.shape[1] == self.num_links
            gs = [[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if array[row, idx] == 1] for row in range(array.shape[0])]
        return GraphSet(gs)

    def zdd_to_array(self, zdd):
        array = [[1 if (f"{i}_out", f"{j}_in") in g or (f"{j}_in", f"{i}_out") in g else 0 for i, j in self.link_indices] for g in list(zdd)]
        return array

    def compute_cost(self, sim):
        cost = np.array(sim.throughput_list).sum() + np.array(sim.throughput_background_list).sum()
        return cost
    
    def const_edge_idx(self):
        const_edges = [(i, j) for i, j in self.link_indices if i in [0, 3, 7, 8] and j in [0, 3, 7, 8]]
        const_edge_idx = [self.link_indices.index(e) for e in const_edges]
        
        return const_edge_idx
    
    @ray.remote(num_cpus=1)
    def evaluate_candidate(individual, params, original_capacity, seq_interval, link_indices, control_start_time, control_end_time, prediction_end_time, time_steps, sim_plant):
        params.max_boundary_capacity = copy.deepcopy(original_capacity)
        sim = copy.deepcopy(sim_plant)
        control_sequence = individual.reshape((time_steps, len(link_indices)))
        for _ in range(control_start_time, prediction_end_time):
            if (sim.step % seq_interval == 0) and (control_start_time <= sim.step < control_end_time):
                sim.params.max_boundary_capacity = copy.deepcopy(original_capacity)
                contraflow_idx = control_sequence[(sim.step // seq_interval) - (control_start_time // seq_interval)]
                for i, j in [link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
                    sim.params.max_boundary_capacity[i, j] *= params.contra_ratio
                    sim.params.max_boundary_capacity[j, i] *= (2 - params.contra_ratio)
            sim.step_simulation()
        cost = np.array(sim.throughput_list).sum() + np.array(sim.throughput_background_list).sum()
        return cost
    
    def plot_p(self, p):
        plt.bar(range(len(p)), p)
        plt.bar([i for i, x in enumerate(p) if x == 1], [1 for x in p if x == 1], color='r')
        plt.ylim(0, 1)
        plt.xlabel("Control variable")
        plt.ylabel("Probability") 
        plt.title(f"Generation {self.generation}")
        plt.savefig(f'{self.output_path}/p_distribution/{self.generation}.png')
        plt.close()    
    
    def fully_adaptive_cross_entropy(self, elite_count, Nmin, Nmax, Nunit, d, max_cpu=1):
        assert elite_count > d, "Elite count must be greater than d"
        self.generation = 0
        population_size = Nmin
        dim = self.num_links * (self.time_steps)
        zdd_constraints = self.constraints_ZDD(constraints_csv=None)
        
        p = np.full(dim, 0.1)
        p0_idx = [t * self.num_links + i for t in range(self.time_steps) for i in self.const_edge_idx()]
        p = [0.0 if i in p0_idx else p[i] for i in range(dim)]
        if self.p:
            self.p = self.p[self.time_steps:] + p[:self.time_steps]
            p = self.alpha * np.array(self.p) + (1-self.alpha) * np.array(p)  # warm start
            p = p.tolist()
        p_list = [p]

        converged = False
        best_individual = None
        fitness_record = []
        gamma_record = []   
        best_cost = -np.inf
        population_size_list = []
        
        while converged == False:
            gc.collect()
            population = np.zeros((population_size, self.num_links*(self.time_steps)))
            costs = []
            concat_list = []
            n_sample = 0
            # First transition
            while n_sample < population_size:
                # generate random control variables
                tmp = (np.random.rand(100, self.num_links) < p[:self.num_links]).astype(int)
                # check if the generated control variables satisfy the constraints
                tmp = self.array_to_zdd(tmp) & zdd_constraints[1]
                if len(tmp) > 0:
                    concat_list.extend(self.zdd_to_array(tmp))
                    n_sample += len(tmp)
            population[:,:self.num_links] = np.concatenate(np.array(concat_list)[:population_size], axis=0).reshape(population_size, self.num_links)
            
            invalid_idx = []
            # Second and subsequent transitions
            for t in range(1, self.time_steps):
                start_time = time.time()
                group, indices = np.unique(population[:,((t-1)*self.num_links):(t*self.num_links)], axis=0, return_inverse=True)
                for idx, ind in enumerate(group):
                    arr = []
                    n_sample = 0
                    target_count = np.sum(indices == idx)
                    count = 0
                    while n_sample < target_count:
                        if count > 1000:
                            print("Too many iterations and forced to break")
                            break
                        # generate random control variables
                        tmp = (np.random.rand(100, self.num_links) < p[(t*self.num_links):((t+1)*self.num_links)]).astype(int)
                        # check if the generated control variables satisfy the constraints
                        zdd_tmp = self.array_to_zdd(tmp) & zdd_constraints[t+1] & self.reconf.transition(self.array_to_zdd(ind), self.reconf.constraints)
                        n = min(len(zdd_tmp), target_count - n_sample)
                        if n > 0:
                            arr.extend(self.zdd_to_array(zdd_tmp)[:n])
                            n_sample += n
                        count += 1
                    if n_sample == target_count:
                        population[indices == idx, (t*self.num_links):((t+1)*self.num_links)] = np.concatenate(np.array(arr), axis=0).reshape(target_count, self.num_links)
                    elif n_sample > 0:
                        arr.extend([arr[-1] for _ in range(target_count - n_sample)])
                        population[indices == idx, (t*self.num_links):((t+1)*self.num_links)] = np.concatenate(np.array(arr), axis=0).reshape(target_count, self.num_links)
                    else:
                        invalid_idx.append(idx)
                
            if len(invalid_idx) > 0:
                valid_indices = np.setdiff1d(np.arange(population_size), invalid_idx)
                for idx in invalid_idx:
                    population[idx] = population[random.choice(valid_indices)].copy()
            
            
            # evaluate the population
            costs = []
            # distributed computing
            ray.init(num_cpus=max_cpu)
            tasks = [
                self.evaluate_candidate.remote(
                    individual, self.params, self.original_capacity, self.seq_interval, 
                    self.link_indices, self.control_start_time, self.control_end_time, prediction_end_time, self.time_steps, self.sim
                ) for individual in population
            ]
            costs = ray.get(tasks)
            ray.shutdown()
            
            gen_best_cost = np.max(costs)
            gamma_hat = np.argsort(costs)[-elite_count]
            print(f"Generation {self.generation}, population {population_size}: Best cost = {gen_best_cost}")
            if gen_best_cost > best_cost:
                best_cost = gen_best_cost
                best_individual = population[np.argmax(costs)]
                print(np.array(best_individual).reshape((self.time_steps, self.num_links)))
            
            # choose elite individuals
            elite_indices = np.argsort(costs)[-elite_count:]
            elite_costs = [costs[i] for i in elite_indices]
            elite_population = [population[i] for i in elite_indices]
            
            if (self.generation < d) or (gen_best_cost > fitness_record[-1] and gamma_hat > gamma_record[-1]):
                pass
            else:
                if gen_best_cost == fitness_record[-1] and np.all([elite_costs[-i] == elite_costs[-i-1] for i in range(1, d)]):
                    print("reliable results")
                    break
                else:
                    if population_size == Nmax:
                        if np.all([population_size_list[-i]==population_size_list[-i-1] for i in range(1, d)]):
                            print("Unreliable results")
                            break
                    else:
                        population_size += Nunit      
                        continue   
            
            # update the probability distribution
            p_new = np.mean(elite_population, axis=0)
            p = self.alpha * p_new + (1-self.alpha) * np.array(p)
            p = [0.0 if i in p0_idx else p[i] for i in range(dim)]
            p_list.append(p)
            self.generation += 1
            gamma_record.append(gamma_hat)
            fitness_record.append(gen_best_cost)
            population_size_list.append(population_size)
            population_size = Nmin
        
        best_control_sequence = best_individual.reshape((self.time_steps, self.num_links))
        self.p = p
        return best_control_sequence, best_cost


    def sim_mpc_step(self, best_control_sequence):
        sim = copy.deepcopy(self.sim)
        sim.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
        contraflow_idx = best_control_sequence[0]
        for i, j in [self.link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
            # contraflow implementation
            sim.params.max_boundary_capacity[i, j] *= self.params.contra_ratio
            sim.params.max_boundary_capacity[j, i] *= (2 - self.params.contra_ratio)
        
        for t in range(self.control_start_time, self.control_start_time + self.seq_interval):
            # Add perturbation to the plant
            sim.step_simulation(self.noise_x[t - sim.sim_start_step], self.noise_x_background[t - sim.sim_start_step])
            
        self.sim = copy.deepcopy(sim)
        self.initial_graph = best_control_sequence[0]
        
        
    def sim_no_policy(self):
        self.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
        sim = copy.deepcopy(self.sim)
        sim.output_path = f'../../output/mfd_dynamics/no_policy'
        
        sim.reset()
        for t in range(sim.sim_start_step, sim.sim_end_step):
            # Add perturbation to the plant
            sim.step_simulation(self.noise_x[t - sim.sim_start_step], self.noise_x_background[t - sim.sim_start_step])
        
        sim.plot_accumulation()
        sim.plot_mfd()
        sim.plot_mfd_animation()
        sim.plot_time()
        sim.plot_throughput()
        sim.plot_jam("../../data/processed/zone_polygon.geojson")
        print(f"Cost with no policy: {self.compute_cost(sim)}")
    
    def sim_best_policy(self, best_control_sequence: np.ndarray):
        sim = copy.deepcopy(self.sim)
        sim.output_path = f'../../output/mfd_dynamics/mpc'
        sim.reset()
        self.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
        for t in range(sim.sim_start_step, sim.sim_end_step):
            if ((sim.step % self.seq_interval) == 0) and (sim.step < (sim.sim_start_step + best_control_sequence.shape[0]*60)):
                sim.params.max_boundary_capacity = copy.deepcopy(self.original_capacity)
                contraflow_idx = best_control_sequence[(sim.step // self.seq_interval) - (sim.sim_start_step // self.seq_interval),:]
                for i, j in [self.link_indices[idx] for idx in np.where(contraflow_idx == 1)[0]]:
                    # contraflow implementation
                    sim.params.max_boundary_capacity[i, j] *= self.params.contra_ratio
                    sim.params.max_boundary_capacity[j, i] *= (2 - self.params.contra_ratio)
            
            # Add perturbation to the plant
            sim.step_simulation(self.noise_x[t - sim.sim_start_step], self.noise_x_background[t - sim.sim_start_step])
            
        sim.plot_accumulation()
        sim.plot_mfd()
        sim.plot_mfd_animation()
        sim.plot_time()
        sim.plot_throughput()
        sim.plot_jam("../../data/processed/zone_polygon.geojson")
        
        best_control_sequence = [[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if tmp[idx] == 1] for tmp in best_control_sequence]
        print(f"Cost with best policy: {self.compute_cost(sim)}")
        self.reconf.draw_sequence(best_control_sequence)
        
        

if __name__ == '__main__':
    noise_scale = 1/20 # Scale of perturbation for state variable x per second. noise_scale*x is the standard deviation of perturbation.
    control_horizon = 6 * 60 # min. control horizon
    prediction_horizon = 6 * 60 # min. prediction horizon
    mpc_start_time = 9 * 60
    mpc_end_time = 18 * 60
    seq_interval = 60 # min. step interval for control sequence
    time_steps = int(control_horizon // seq_interval)
    initial_graph = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    elite_count, N_min, N_max, N_unit, d, max_cpu = 5, 40, 100, 30, 3, 2 # for laptop
    
    #elite_count, N_min, N_max, N_unit, d, max_cpu = 10, 400, 1000, 300, 3, 25 # for server
    
    # MPC
    contraflow_graphs = []
    for control_start_time in tqdm(range(mpc_start_time, mpc_end_time, seq_interval)):
        start = time.time()
        control_end_time = control_start_time + control_horizon # min. 
        prediction_end_time = control_start_time + prediction_horizon # min. 
        
        if control_start_time == mpc_start_time:
            params = Parameters()
            params.simulation_start_time = mpc_start_time
            params.simulation_end_time = mpc_end_time+prediction_horizon
            params.T_all = params.simulation_end_time - params.simulation_start_time
            ce = CrossEntropy(params, initial_graph, time_steps, seq_interval, control_start_time, control_end_time, prediction_end_time, noise_scale)
            ce.sim_no_policy()
        else:
            ce.control_start_time= control_start_time
            ce.control_end_time = control_end_time
            ce.prediction_end_time = prediction_end_time
            
        best_sequence, best_cost = ce.fully_adaptive_cross_entropy(elite_count=elite_count, Nmin=N_min, Nmax=N_max, Nunit=N_unit, d=d, max_cpu=max_cpu)
        ce.sim_mpc_step(best_sequence)
        contraflow_graphs.append(best_sequence[0])
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        print("Optimal Control Sequence (each row corresponds to a time step, each column to a link):")
        print(best_sequence)
        print(f"Best cost: {best_cost}")
    
    pd.DataFrame(contraflow_graphs).to_csv(f'{ce.output_path}/contraflow_graphs.csv', index=False, header=False)
        
    ce.sim_best_policy(np.array(contraflow_graphs))
    

