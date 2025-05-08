import numpy as np
import networkx as nx
import pandas as pd
import os
import sys
import json
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from data.mfd_zoning import MFD_Zoning

class Parameters:
    def __init__(self):
        #mfd_dynamics
        mfd_zoning = MFD_Zoning(network_path="../../data/raw/network.geojson", cluster_path="../../data/raw/zoning.csv")
        mfd_params = pd.read_csv("../../data/processed/fitted_scale_factors.csv")
        avg_trip_length = pd.read_csv("../../data/processed/average_trip_lengths.csv")
        shelter_list = pd.read_csv("../../data/processed/shelter_list.csv")
        boundary_capacity_file = "../../data/processed/boundary_capacity.csv"
        parking_success_file = "../../data/processed/parking_params.csv"
        self.num_zones = len(mfd_zoning.clusters)
        
        self.demand = "demand_36h"
        self.demand_variation = "93_max" #"67_min", #"75_0.25", #"46_mean" #"16_0.75" #"38_0.8", #"78_0.85", #"71_0.9", #"53_0.95", #"93_max", # "normal"
        self.background_ratio = 0.8
        self.contra_ratio = 2.0
        self.green_split = 0.5
        self.simulation_start_time = 9 * 60 # Simulation start time, min
        self.simulation_end_time = 19 * 60 # Simulation end time, min # 48 * 60
        self.T_all = self.simulation_end_time - self.simulation_start_time # Total simulation time (number of time periods for OD), min
        self.sampling_time = 1 # State update interval in minutes
        
        od_file = f"../../data/processed/{self.demand}/{self.demand_variation}/evac_od.csv"
        if self.demand_variation == "normal":
            od_file = None
        
        background_traffic_file = f"../../data/processed/{self.demand}/{self.demand_variation}/normal_od.csv"
        if self.demand_variation == "normal":
            background_traffic_file = "../../data/processed/normal_od_all.csv"
        #background_traffic_file = None
        
        # Zone configuration
        if os.path.exists("../../data/processed/adj_matrix.csv"):
            self.adj_matrix = np.loadtxt("../../data/processed/adj_matrix.csv", delimiter=',')
        else:
            adjacent_clusters = mfd_zoning.find_adjacent_clusters()
            self.adj_matrix = np.array([[1 if j in adjacent_clusters[i] else 0 for j in range(self.num_zones)] for i in range(self.num_zones)])
            np.savetxt("../../data/processed/adj_matrix.csv", self.adj_matrix, delimiter=',')
        # Path enumeration (k-shortest path)
        if os.path.exists("../../data/processed/path_dict.json"):
            with open("../../data/processed/path_dict.json", "r") as f:
                loaded_data = json.load(f)
            self.path_dict = {eval(k): [eval(path) for path in v] for k, v in loaded_data.items()}
        else:
            self.path_dict = mfd_zoning.find_paths(avg_trip_length)
            json_data = {str(k): [str(path) for path in v] for k, v in self.path_dict.items()}
            with open("../../data/processed/path_dict.json", "w") as f:
                json.dump(json_data, f)

        self.path_update_interval = 1  # Path update interval, min
        self.compliance_rate = 1.0  # dispatching compliance rate
        # MFD parameters
        self.A = np.array(mfd_params['coef1']) / 60 # MFD parameter veh*km/h -> veh*km/min
        self.B = np.array(mfd_params['coef2']) / 60 # MFD parameter veh*km/h -> veh*km/min
        self.C = np.array(mfd_params['coef3']) / 60 # MFD parameter veh*km/h -> veh*km/min
        self.N_jam =  np.array(mfd_params['N_jam']) # Maximum capacity of horizontal network in zone, veh
        
        # outer zone
        self.outer_zones = [0, 3, 7, 8]
        
        # Average trip length
        self.L_m = avg_trip_length['avg_trip_length_km'].values  # Average trip length in zone i, km
        self.L_o = np.zeros((self.num_zones, self.num_zones))  # Average trip length between zones i and j, km
        for i in range(self.num_zones):
            for j in range(self.num_zones):
                self.L_o[i, j] = min(self.path_dict[(i,j)], key=lambda x: x[1])[1] if i != j else self.L_m[i]
        
        # Route choice probability
        self.theta_o = self.k_shortest_theta() #self.shortest_path_theta() #self.generate_theta(self.num_zones)
        self.theta_d = self.theta_o.copy() # 3D array [i,j,k]. Proportion of trips from zone i to j when dispatched to zone k
        # Parameters for evacuation success rate
        parking_success_params = pd.read_csv(parking_success_file)
        self.Ap = parking_success_params['Ap'].values
        self.eta1 = parking_success_params['eta1'].values
        self.eta2 = parking_success_params['eta2'].values
        self.eta3 = parking_success_params['eta3'].values
        # Vertical and horizontal evacuation capacity
        self.Cap_i = shelter_list.groupby(['zone_id'])['capacity'].sum().values  # Vertical shelter capacity in zone i, veh
        #self.Cap_i *= 100
        self.max_boundary_capacity = np.loadtxt(boundary_capacity_file, delimiter=',') * self.green_split # Maximum boundary capacity for each zone, veh/min
        self.alpha = 0.64 # Parameter used for boundary capacity calculation
        
        # OD matrix
        if od_file is not None:
            self.Q, self.od_df = self.load_od_matrix(od_file)  # Load OD matrix from CSV
        else:
            self.Q = np.zeros((self.num_zones, self.num_zones, int(48*60/self.sampling_time)))
            self.od_df = pd.DataFrame()
            
        if background_traffic_file is not None:
            self.Q_background, _ = self.load_od_matrix(background_traffic_file)  # Load background OD matrix from CSV
        else:
            self.Q_background = np.zeros((self.num_zones, self.num_zones, int(48*60/self.sampling_time)))
        
        if self.demand_variation != "normal" and not os.path.exists(f'../../data/processed/{self.demand}/{self.demand_variation}/od_distribution_evac.geojson'):
            self.plot_od("../../data/processed/zone_polygon.geojson")
        
        # For visualization
        color_list = [
            'Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Purple', 'Pink',
            'Lime',  'Gray', 'Cyan', 'Magenta', 'Brown',
            'Maroon', 'Navy', 'Olive', 'Teal', 'Aqua', 'Silver', 'Coral', 'Black'
        ]
        colors = color_list[:self.num_zones]
        self.color_dict = dict(zip(range(self.num_zones), colors))
    
    # 3D array [i,j,k]. Proportion of trips from zone i to j when targeting zone k. sum(over j) theta_i,k must = 1
    def generate_theta(self, num_zones):
        # Dirichlet distribution parameters
        alpha = np.ones(num_zones)  # When all parameters are 1, distribution is uniform
        # Initialize theta[i, j, k]
        theta = np.zeros((num_zones, num_zones, num_zones))
        for i in range(num_zones):
            for k in range(num_zones):
                # Generate random vector following Dirichlet distribution
                theta[i, :, k] = np.random.dirichlet(alpha)
        return theta
    
    def shortest_path_theta(self):
        G = nx.from_numpy_array(self.adj_matrix)
        theta = np.zeros((self.num_zones, self.num_zones, self.num_zones))
        
        for k in range(self.num_zones):
            for i in range(self.num_zones):
                shortest_paths = nx.single_source_shortest_path(G, i)
                for j in range(self.num_zones):
                    if j in shortest_paths[k]:
                        theta[i, j, k] = 1
        
        return theta
    
    def k_shortest_theta(self):
        theta = np.zeros((self.num_zones, self.num_zones, self.num_zones))
        # assign probability of choosing next zone based on softmax of average trip length
        for k in range(self.num_zones):
            for i in range(self.num_zones):
                if i != k:
                    k_shortest_path = self.path_dict[(i, k)]
                    for path, cost in k_shortest_path:
                        j = path[1] # Next zone after i in kth shortest path from i to k
                        # Add route choice probability based on avg_trip_length
                        theta[i, j, k] += np.exp(-cost)
        # normalize (softmax)
        for k in range(self.num_zones):
            for i in range(self.num_zones):
                if np.sum(theta[i, :, k]) != 0:
                    theta[i, :, k] /= np.sum(theta[i, :, k])
                
        return theta
    
    def load_od_matrix(self, od_file):
        df = pd.read_csv(od_file)
        df['time'] = df['time'] // 60  # Convert time to minutes
        Q = np.zeros((self.num_zones, self.num_zones, int(48*60/self.sampling_time))) # OD matrix for entire period

        for _, row in df.iterrows():
            time = int(row['time'])
            origin = int(row['origin'])
            destination = int(row['destination'])
            value = row['value']
            Q[origin, destination, time] += value
        '''
        # Set trips between outer zones to 0
        for i in self.outer_zones:
            for j in self.outer_zones:
                Q[i, j, :] = 0
        '''
        # Set internal trips within outer zones to 0
        for i in self.outer_zones:
            Q[i, i, :] = 0
                
        return Q, df
    
    def plot_od(self, polygon_path):
        '''
        Add OD distribution on the geojson file of zone polygon
        '''    
        def plot_od_line(gdf, output_path, mode):
            # Calculate centroid of each polygon
            gdf["centroid"] = gdf.geometry.centroid
            # Create NetworkX graph
            G = nx.DiGraph()  # Directed graph (to represent OD relationships)
            # Add nodes (polygon centroids as nodes)
            pos = {}  # Store node coordinates for drawing
            for _, row in gdf.iterrows():
                centroid = (row.centroid.x, row.centroid.y)
                G.add_node(row.cluster, pos=centroid)
                pos[row.cluster] = centroid

            # Register OD traffic volumes as NetworkX edges
            for i in range(self.num_zones):
                for j in range(self.num_zones):
                    G.add_edge(i, j, weight=gdf.loc[gdf['cluster'] == i, f'cluster_{j}'].values[0])
            # Draw
            fig, ax = plt.subplots(figsize=(10, 8))
            # Set colormap
            cmap = plt.cm.Reds if mode=='demand' else plt.cm.RdBu_r
            norm = mcolors.Normalize(vmin=gdf[mode].min(), vmax=gdf[mode].max())
            gdf.plot(column=mode, cmap=cmap, linewidth=0.5, edgecolor="black", alpha=0.7, ax=ax)
            # Add colorbar
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Set empty array (matplotlib requirement)

            # Add colorbar to plot
            cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
            if mode == 'demand':
                cbar.set_label("Total demand", fontsize=12)
            else:
                cbar.set_label("Departed - arrived", fontsize=12)
            # Draw nodes (centroids)
            nx.draw_networkx_nodes(G, pos, node_size=50, node_color="red", alpha=0.7)
            # Draw edges (OD relationships) (adjust width and color based on OD volume)
            edges = G.edges(data=True)
            weights = [d["weight"] for _, _, d in edges]
            nx.draw_networkx_edges(
                G,
                pos,
                width=[w / max(weights) * 5 for w in weights],  # Adjust line width based on flow
                edge_color="black",
                alpha=0.6,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.1",
            )
            # Hide axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            #plt.title("OD Connection Map", fontsize=14)
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
            plt.close()
        
        polygon = gpd.read_file(polygon_path)
        
        polygon_evac = polygon.copy()
        for i in range(self.num_zones):
            for j in range(self.num_zones):
                polygon_evac.loc[polygon_evac['cluster'] == i, f'cluster_{j}'] = self.Q[i,j,:].sum()
        polygon_evac["demand"] = polygon_evac["cluster"].apply(lambda x: self.Q[x, :, :].sum()) #+ self.Q[:, x, :].sum())
        polygon_evac["dep-arr"] = polygon_evac["cluster"].apply(lambda x: self.Q[x, :, :].sum() - self.Q[:, x, :].sum())
        polygon_evac.to_file(f'../../data/processed/{self.demand}/{self.demand_variation}/od_distribution_evac.geojson', driver='GeoJSON')
        plot_od_line(polygon_evac, f'../../data/processed/{self.demand}/{self.demand_variation}/demand_distribution_evac.png', mode='demand')
        plot_od_line(polygon_evac, f'../../data/processed/{self.demand}/{self.demand_variation}/concentration_evac.png', mode='dep-arr')
        
        polygon_normal = polygon.copy()
        for i in range(self.num_zones):
            for j in range(self.num_zones):
                polygon_normal.loc[polygon_normal['cluster'] == i, f'cluster_{j}'] = self.Q_background[i,j,:].sum()
        polygon_normal["demand"] = polygon_normal["cluster"].apply(lambda x: self.Q_background[x, :, :].sum())# + self.Q_background[:, x, :].sum())
        polygon_normal["dep-arr"] = polygon_normal["cluster"].apply(lambda x: self.Q_background[x, :, :].sum() - self.Q_background[:, x, :].sum())
        polygon_normal.to_file(f'../../data/processed/{self.demand}/{self.demand_variation}/od_distribution_normal.geojson', driver='GeoJSON')
        plot_od_line(polygon_normal, f'../../data/processed/{self.demand}/{self.demand_variation}/demand_distribution_normal.png', mode='demand')
        plot_od_line(polygon_normal, f'../../data/processed/{self.demand}/{self.demand_variation}/concentration_normal.png', mode='dep-arr')
        
        polygon_all = polygon.copy()
        for i in range(self.num_zones):
            for j in range(self.num_zones):
                polygon_all.loc[polygon_all['cluster'] == i, f'cluster_{j}'] = self.Q[i,j,:].sum() + self.Q_background[i,j,:].sum()
        polygon_all["demand"] = polygon_all["cluster"].apply(lambda x: self.Q[x, :, :].sum() + self.Q_background[x, :, :].sum())# + self.Q[:, x, :].sum() + self.Q_background[:, x, :].sum())
        polygon_all["dep-arr"] = polygon_all["cluster"].apply(lambda x: self.Q[x, :, :].sum() + self.Q_background[x, :, :].sum() - self.Q[:, x, :].sum() - self.Q_background[:, x, :].sum())
        polygon_all.to_file(f'../../data/processed/{self.demand}/{self.demand_variation}/od_distribution_all.geojson', driver='GeoJSON')
        plot_od_line(polygon_all, f'../../data/processed/{self.demand}/{self.demand_variation}/demand_distribution_all.png', mode='demand')
        plot_od_line(polygon_all, f'../../data/processed/{self.demand}/{self.demand_variation}/concentration_all.png', mode='dep-arr')
        