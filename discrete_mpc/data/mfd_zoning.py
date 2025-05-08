import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from itertools import islice


class MFD_Zoning:
    def __init__(self, network_path, cluster_path):
        self.network_path = network_path
        self.network = None
        self.G = None
        self.S = None
        self.edges = {}
        
        # load_network
        self.network = gpd.read_file(self.network_path)
        cluster_df = pd.read_csv(cluster_path, encoding='utf-8')
        self.clusters = [cluster_df[column].dropna().to_list() for column in cluster_df.columns]
        self.cluster2edge = {cluster_id: cluster_df[column].dropna().to_list() for cluster_id, column in enumerate(cluster_df.columns)}
        self.edge2cluster = {edge_id: cluster_id for cluster_id, cluster in enumerate(self.clusters) for edge_id in cluster}
        
        # def extract_edges
        for _, row in self.network.iterrows():
            coords = row['geometry'].coords
            if len(coords) > 1:
                start = coords[0]
                end = coords[-1]
                edge = (tuple(start), tuple(end))
                self.edges[row['id']] = edge
        print("Number of edges in the original SUMO graph:", len(self.edges))
        
        self.output_dir = "../../output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.create_graph()
    
    def create_graph(self):
        self.G = nx.MultiGraph()
        for edge_id, (start, end) in self.edges.items():
            self.G.add_edges_from(
                [(start, end)], id=edge_id
            )
        self._remove_small_clusters(edge_threshold=100)
        self._create_line_graph()
        self.id2node_dict = {edge_id: (s, t, x) for (s, t, x), edge_id in self.S.nodes(data='id')}
        
        print("Created graphs successfully")
    
    def _remove_small_clusters(self, edge_threshold):
        connected_components = list(nx.connected_components(self.G))
        nodes_to_remove = []
        for component in connected_components:
            subgraph = self.G.subgraph(component)
            if subgraph.number_of_edges() <= edge_threshold:
                nodes_to_remove.extend(list(component))
        self.G.remove_nodes_from(nodes_to_remove)
        print("Number of nodes in the networkx graph:", len(list(self.G.nodes)))
        print("Number of edges in the networkx graph:", len(list(self.G.edges)))
        
    def _create_line_graph(self):
        '''
        Create a line graph from a graph G. Line graph is a graph where the nodes represent the edges of the original graph G.
        
        Parameters
        ----------
        G : nx.Graph
            A graph object.
        
        Returns
        -------
        S : nx.Graph
            A line graph object.
        '''
        self.S = nx.line_graph(self.G)
        for s, t, x in self.G.edges:
            self.S.nodes[(s, t, x)]['id'] = self.G.edges[(s, t, x)]['id']
        
        
    def calculate_all_pairs_shortest_paths(self):
        return dict(nx.all_pairs_shortest_path_length(self.G))
    
    def find_adjacent_clusters(self):
        '''
        Find the adjacent clusters of each cluster in the graph
        
        Parameters
        ----------
        clusters: list
            The list of clusters of the graph (each cluster is a list of nodes)
        
        Returns
        -------
        cluster_adjacency: dict
            The dictionary of the adjacent clusters of each cluster
        '''
        cluster_adjacency = {}            
        for i, cluster in enumerate(self.clusters):
            cluster_adjacency[i] = set()
            for edge_id in cluster:
                for neighbor in self.S.neighbors(self.id2node_dict[edge_id]):
                    for j, other_cluster in enumerate(self.clusters):
                        if j != i and self.S.nodes[neighbor].get('id') in other_cluster:
                            cluster_adjacency[i].add(j)
        
        return cluster_adjacency
    
    def find_paths(self, trip_lenth_file: pd.DataFrame = None):
        '''
        Find all paths between each pair of clusters
        
        Parameters
        ----------
        clusters: list
            The list of clusters of the graph (each cluster is a list of nodes)
        trip_lenth_file: pd.DataFrame
            The file containing the trip lengths within each cluster
        
        Returns
        -------
        path_dict: dict
            The dictionary of all paths between each pair of clusters
        '''
        adjacent_clusters = self.find_adjacent_clusters()
        num_zones = len(self.clusters)
        adj_matrix = np.array([[1 if j in adjacent_clusters[i] else 0 for j in range(num_zones)] for i in range(num_zones)])
        zone_graph = nx.from_numpy_array(adj_matrix)
        if trip_lenth_file is not None:
            # 重みを設定
            for i, edge in enumerate(zone_graph.edges):
                s,t = edge
                zone_graph.edges[edge]['weight'] = trip_lenth_file["avg_trip_length_km"].values[s] + trip_lenth_file["avg_trip_length_km"].values[t]
            
        
        def k_shortest_paths(G, source, target, k, weight=None):
            return list(
                islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
                )
        
        path_dict = {}
        for i, s_node in enumerate(zone_graph.nodes):
            for j, t_node in enumerate(zone_graph.nodes):
                if i != j:
                    path_dict[(i, j)] = []
                    for path in k_shortest_paths(zone_graph, s_node, t_node, 5):
                        path_dict[(i, j)].append((path, nx.path_weight(zone_graph, path, weight='weight')))
        
        return path_dict
