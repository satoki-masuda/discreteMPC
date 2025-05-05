'''
初期グラフから目標グラフへ一定時間内に遷移可能な経路を求めるアルゴリズム
ZDD, BFS, DFS, A*のアルゴリズムを実装
'''
import shutil
import psutil
import heapq
import time
import math
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
import itertools
from tqdm import tqdm
import threading
import random
from functools import reduce
import operator
os.environ["RAY_DEDUP_LOGS"] = "0"
import ray
from graphillion import setset, GraphSet
from parameters_ndp import Parameters

class ReconfigZDD():
    def __init__(self, params, initial_graph, target_graph, k, constraints_csv=None, output_path=None):
        self.params = params
        self.output_path = output_path
        self.k = k
        num_zones = params.num_zones
        if output_path is not None:
            # 既にフォルダが存在する場合は中身を削除
            if os.path.exists(f'{self.output_path}'):
                shutil.rmtree(f'{self.output_path}')
            os.makedirs(f'{self.output_path}')
            os.makedirs(f'{self.output_path}/seq')
                
        self.pos = {0: (1,5.5), 1: (3,2), 2: (5.5,2), 3:(8,4), 4:(2.5,5), 5:(6,4.5), 6:(3.5, 4.25), 7:(3, 7.5), 8:(6,7)}
        
        # リンクの総数を隣接行列に基づいて計算
        self.link_indices = [(i, j) for i in range(num_zones) for j in range(num_zones) if params.adj_matrix[i, j] == 1]
        self.initial_link_indices = [(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if initial_graph[idx] == 1]
        self.target_link_indices = [(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if target_graph[idx] == 1]
        
        self.constraints_csv = constraints_csv
        self.make_graph()
        
    def make_graph(self):
        self.G = nx.Graph()
        self.G_edges = set()
        # 各エッジを入ノードと出ノードに分けて追加
        for u, v in self.link_indices:
            self.G.add_edge(f"{u}_out", f"{v}_in")
            self.G_edges.add((f"{u}_out", f"{v}_in"))
        # エッジリストからグラフを作成
        GraphSet.set_universe(self.G_edges)

    def make_constraints(self):
        #GraphSet.set_universe(self.G.edges())
        constraints = GraphSet({})
        
        # 相対するエッジが同時に存在しない
        opposite_edges = [[(f"{i}_out", f"{j}_in"), (f"{j}_out", f"{i}_in")] for i, j in self.link_indices]
        for edges in opposite_edges:
            without_opposite_graph = ~GraphSet({}).including(edges)
            constraints = constraints.intersection(without_opposite_graph)
        
        # ゾーンから出られない状態は発生しない
        all_in_edges = [[(f"{j}_out", f"{i}_in") for j in range(self.params.num_zones) if self.params.adj_matrix[j, i] == 1] for i in range(self.params.num_zones)]
        for edges in all_in_edges:
            without_in_edges = ~GraphSet({}).including(edges)
            constraints = constraints.intersection(without_in_edges)
            
        # ゾーン1,2,4,5,6に関するリンクのみを考慮
        const_edges = [[(f"{i}_out", f"{j}_in")] for i, j in self.link_indices if i in [0, 3, 7, 8] and j in [0, 3, 7, 8]]
        #const_edges += [[(f"{i}_out", f"{j}_in")] for i, j in [(0, 4), (4, 0), (5, 6), (6, 5), (5, 8), (8, 5)]]
        for edges in const_edges:
            without_const_edges = ~GraphSet({}).including(edges)
            constraints = constraints.intersection(without_const_edges)
        
        # 制約条件のcsvファイルから制約条件をみたさないグラフを読み込む
        if self.constraints_csv is not None:
            constraints_gs = GraphSet([[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if data[f'link{idx}'] == 1] for _, data in self.constraints_csv.iterrows()])
            constraints = constraints.difference(constraints_gs) # 制約条件を満たさないグラフを削除
        
        self.constraints = constraints
        print(f"Number of search space: {len(constraints)}")
        print(f"Number of eliminated graphs: {len(GraphSet({})) - len(constraints)}")
        
    def transition(self, sz, constraints):
        return (sz.add_some_edge() | sz.remove_some_edge() | sz.remove_add_some_edges() | sz) & constraints

    def _get_reconf_set(self, setset_seq, s, t, search_space):
        '''
        終端状態から初期状態まで逆向きに辿って，全ての遷移過程を取得するための関数 (目標グラフから後ろ向きに探索)
        '''
        current_set = GraphSet([t])
        reconf_set = [current_set]
        # 終端状態の一つ前のグラフ集合から初期状態まで逆向きに辿る
        for i in range(len(setset_seq) - 2, -1, -1):
            # 初期グラフから遷移可能なグラフ集合setset_seq[i]の中で，目標グラフから遷移可能なグラフ集合next_ssに含まれるグラフを選択
            next_ss = self.transition(current_set, setset_seq[i])
            if next_ss:
                current_set = next_ss
                reconf_set.append(current_set)
            else:
                print("No transition")
                return []
        return reconf_set[::-1]

    def _get_seq_all(self, setset_seq, s, t, search_space):
        '''
        終端状態から初期状態まで逆向きに辿って，全ての遷移過程を取得するための関数 (目標グラフから後ろ向きに探索)
        '''
        reconf_size = self._get_reconf_set(setset_seq, s, t, search_space)
        level_list = [len(s) for s in reconf_size]
        print("Possible intermediate states: ", level_list)
        
        reconf_seq_all = []
        def dfs(current_grpah, path, level, count):
            if level == 0:
                reconf_seq_all.append(path + [current_grpah])
                count += 1
                print(f"Leaf node reached! Current count: {count}")
                return count
            local_count = count
            for next_graph in self.transition(current_grpah, setset_seq[level-1]):
                next_graph = GraphSet([next_graph])
                local_count = dfs(next_graph, path + [next_graph], level - 1, local_count)
            return local_count
        reconf_seq = [GraphSet([t])]
        total_count = dfs(GraphSet([t]), reconf_seq, len(setset_seq) - 1, count=0)
        print(f"Total leaf node count: {total_count}")
        return reconf_seq_all
    
    def _get_random_seq(self, setset_seq, s, t, search_space, num_samples=100):
        '''
        終端状態から初期状態まで逆向きに辿って，全ての遷移過程を取得するための関数 (目標グラフから後ろ向きに探索)
        '''
        reconf_size = self._get_reconf_set(setset_seq, s, t, search_space)
        level_list = [len(s) for s in reconf_size]
        print("Possible intermediate states: ", level_list)
        
        reconf_seq_list = [[] for _ in range(num_samples)]
        for sample in range(num_samples):
            reconf_seq = [t]
            current_set = t
            # 終端状態の一つ前のグラフ集合から初期状態まで逆向きに辿る
            for i in range(len(setset_seq) - 2, -1, -1):
                sz = GraphSet([current_set])
                next_ss = self.transition(sz, setset_seq[i])
                for gs in next_ss.rand_iter():
                    current_set = gs
                    break
                reconf_seq.append(current_set)
            reconf_seq_list[sample] = reconf_seq[::-1]
            
        return reconf_seq_list
        
        
    def get_reconf_seq(self, s, t, search_space):
        '''
        目標グラフまでの遷移過程を取得するための関数 (初期グラフから前向きに探索)
        - s: 初期グラフ
        - t: 目標グラフ
        - search_space: 探索空間
        - k: 何ステップ先まで探索するか
        '''
        setset_seq = [] # 初期状態から遷移できるグラフの集合
        setset_seq.append(GraphSet([s])) # 初期状態を追加

        # 終端時刻になるまで遷移を続ける
        while len(setset_seq) <= self.k:
            # グラフから1本の辺を削除し，1本の辺を追加して得られるグラフの集合
            next_ad = setset_seq[-1].add_some_edge()
            next_rm = setset_seq[-1].remove_some_edge()
            next_tj = setset_seq[-1].remove_add_some_edges()
            next_ss = (next_ad | next_rm | next_tj | setset_seq[-1]) & search_space
            setset_seq.append(next_ss) # next_ssを追加
            print(len(setset_seq), len(next_ss))
            
        if t in next_ss: # next_ssにtが含まれていれば，_get_seqで遷移過程を取得して終了
            print("遷移系列あり")
            #return self._get_random_seq(setset_seq, s, t, search_space, num_samples=100)
            #return self._get_seq_all(setset_seq, s, t, search_space)
            return self._get_reconf_set(setset_seq, s, t, search_space)
        else:
            print("遷移系列なし")
            return []
    
    def draw_sequence(self, reconf_seq):
        all_edges = []
        for edge in self.G.edges():
            u, v = edge
            in_node = int(u.split("_")[0]) if "in" in u.split("_") else int(v.split("_")[0])
            out_node = int(v.split("_")[0]) if "out" in v.split("_") else int(u.split("_")[0])
            all_edges.append((out_node, in_node))
            all_edges.append((in_node, out_node))

        for i, g in enumerate(reconf_seq):
            #print(i, g)
            graph = nx.DiGraph()
            graph.add_nodes_from([i for i in range(self.params.num_zones)])
            contra_edges = []
            
            for e in g:
                # inとついている方がinノード
                u, v = e
                in_node = int(u.split("_")[0]) if "in" in u.split("_") else int(v.split("_")[0])
                out_node = int(v.split("_")[0]) if "out" in v.split("_") else int(u.split("_")[0])
                contra_edges.append((out_node, in_node))
            
            nx.draw_networkx_nodes(graph, self.pos, node_color="lightblue", node_size=300)
            
            # 全エッジをweightに基づいて描画
            for edge in all_edges:
                width = 1  # エッジが `edge_labels` にない場合のデフォルト太さ
                nx.draw_networkx_edges(graph, self.pos, edgelist=[edge], edge_color="lightgrey", width=width,
                                    arrowstyle="->", connectionstyle="arc3,rad=0.1", style="-")
                if edge in contra_edges:
                    nx.draw_networkx_edges(graph, self.pos, edgelist=[edge], edge_color="red", width=width,
                                        arrowstyle="->", connectionstyle="arc3,rad=0.1")
            # ノードとラベルの描画
            nx.draw_networkx_labels(graph, self.pos, font_size=12, font_color="black")
            #nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, bbox=dict(facecolor="white", edgecolor="none", alpha=0.0), label_pos=0.65)
            plt.title(f"Step {i+1}")
            # 描画表示
            plt.axis("off")
            if self.output_path is not None:
                plt.savefig(f"{self.output_path}/seq/step_{i+1}.png")
            plt.close()
    
    def draw_initial_target_graphs(self):
        all_edges = []
        for edge in self.G.edges():
            u, v = edge
            in_node = int(u.split("_")[0])
            out_node = int(v.split("_")[0])
            all_edges.append((out_node, in_node))
            all_edges.append((in_node, out_node))

        graph = nx.DiGraph()
        graph.add_nodes_from([i for i in range(self.params.num_zones)])
        pre_edges = []
        post_edges = []

        for e in self.G.edges():
            # inとついている方がinノード
            u, v = e
            in_node = int(u.split("_")[0]) if "in" in u.split("_") else int(v.split("_")[0])
            out_node = int(v.split("_")[0]) if "out" in v.split("_") else int(u.split("_")[0])
            if (u,v) in self.initial_link_indices or (v,u) in self.initial_link_indices:
                pre_edges.append((out_node, in_node))
            if (u,v) in self.target_link_indices or (v,u) in self.target_link_indices:
                post_edges.append((out_node, in_node))
            
        nx.draw_networkx_nodes(graph, self.pos, node_color="lightblue", node_size=300)
        # 全エッジのデフォルトの色と太さで描画
        nx.draw_networkx_edges(graph, self.pos, edgelist=all_edges, edge_color="black", width=1,
                                arrowstyle="->",connectionstyle="arc3,rad=0.1")
        # 選択したエッジだけ色と太さを変更
        nx.draw_networkx_edges(graph, self.pos, edgelist=pre_edges, edge_color="blue", width=3,
                                arrowstyle="->",connectionstyle="arc3,rad=0.1")
        # ノードとラベルの描画
        nx.draw_networkx_labels(graph, self.pos, font_size=12, font_color="black")
        plt.title(f"Normal time")
        # 描画表示
        plt.axis("off")
        if self.output_path is not None:
            plt.savefig(f"{self.output_path}/initial_graph.png")
        plt.close()

            
        nx.draw_networkx_nodes(graph, self.pos, node_color="lightblue", node_size=300)
        # 全エッジのデフォルトの色と太さで描画
        nx.draw_networkx_edges(graph, self.pos, edgelist=all_edges, edge_color="black", width=1,
                                arrowstyle="->",connectionstyle="arc3,rad=0.1")
        # 選択したエッジだけ色と太さを変更
        nx.draw_networkx_edges(graph, self.pos, edgelist=post_edges, edge_color="red", width=3,
                                arrowstyle="->",connectionstyle="arc3,rad=0.1")
        # ノードとラベルの描画
        nx.draw_networkx_labels(graph, self.pos, font_size=12, font_color="black")
        plt.title(f"Emergency time")
        # 描画表示
        plt.axis("off")
        if self.output_path is not None:
            plt.savefig(f"{self.output_path}/target_graph.png")
        plt.close()
    
    def reconfiguration(self):
        self.make_constraints()
        #self.constraints = GraphSet({})
        reconf_seq = self.get_reconf_seq(self.initial_link_indices, self.target_link_indices, self.constraints)
        return reconf_seq
    

class ReconfigZDD_MPC(ReconfigZDD):
    def __init__(self, params, initial_graph, k, constraints_csv=None, output_path=None):
        super().__init__(params, initial_graph, initial_graph, k, constraints_csv, output_path)
    
    def get_reconf_seq(self, s, search_space):
        '''
        目標グラフまでの遷移過程を取得するための関数 (初期グラフから前向きに探索)
        - s: 初期グラフ
        - t: 目標グラフ
        - search_space: 探索空間
        - k: 何ステップ先まで探索するか
        '''
        setset_seq = [] # 初期状態から遷移できるグラフの集合
        setset_seq.append(GraphSet([s])) # 初期状態を追加

        # 終端時刻になるまで遷移を続ける
        while len(setset_seq) <= self.k:
            # グラフから1本の辺を削除し，1本の辺を追加して得られるグラフの集合
            next_ad = setset_seq[-1].add_some_edge()
            next_rm = setset_seq[-1].remove_some_edge()
            next_tj = setset_seq[-1].remove_add_some_edges()
            next_ss = (next_ad | next_rm | next_tj | setset_seq[-1]) & search_space
            setset_seq.append(next_ss) # next_ssを追加
            #print(len(setset_seq), len(next_ss))
            
        return setset_seq
    
    def reconfiguration(self):
        self.make_constraints()
        #self.constraints = GraphSet({})
        reconf_seq = self.get_reconf_seq(self.initial_link_indices, self.constraints)
        return reconf_seq

class ReconfigBFS(ReconfigZDD):
    def __init__(self, params, initial_graph, target_graph, k, constraints_csv=None, output_path=None):
        super().__init__(params, initial_graph, target_graph, k, constraints_csv, output_path)
        self.initial_link_indices = [(i, j) for idx, (i, j) in enumerate(self.link_indices) if initial_graph[idx] == 1]
        self.target_link_indices = [(i, j) for idx, (i, j) in enumerate(self.link_indices) if target_graph[idx] == 1]
        #self.make_graph()
        
    def make_graph(self):
        # 入ノードと出ノードに分けずにエッジを追加
        self.G = nx.DiGraph()
        for u, v in self.link_indices:
            self.G.add_edge(u, v)
        
    def make_constraints(self):
        self.constraints = []
        # 相対するエッジが同時に存在しない
        opposite_edges = [[(i, j), (j, i)] for i, j in self.link_indices]
        self.constraints.extend(opposite_edges)
        
        # ゾーンから出られない状態は発生しない
        all_in_edges = [[(j, i) for j in range(self.params.num_zones) if self.params.adj_matrix[j, i] == 1] for i in range(self.params.num_zones)]
        self.constraints.extend(all_in_edges)
        
        # ゾーン1,2,4,5,6に関するリンクのみを考慮
        const_edges = [[(i, j)] for i, j in self.link_indices if i in [0, 3, 7, 8] and j in [0, 3, 7, 8]]
        self.constraints.extend(const_edges)
        
        # 制約条件のcsvファイルから制約条件を読み込む
        if self.constraints_csv is not None:
            self.constraints.extend([[(i, j) for idx, (i, j) in enumerate(self.link_indices) if data[f'link{idx}'] == 1] for _, data in self.constraints_csv.iterrows()])
        
    def transition(self, graph):
        """
        現在のグラフから遷移可能なグラフの集合を返す
        - add_some_edge: 1本の辺を追加
        - remove_some_edge: 1本の辺を削除
        - remove_add_some_edges: 1本の辺を削除し、1本の辺を追加
        - 現状維持も含む
        """
        new_states = [graph]
        
        # add_some_edge
        for edge in self.link_indices:
            if edge not in graph.edges:
                modified_graph = graph.copy()
                modified_graph.add_edge(*edge)
                if self._satisfies_constraints(modified_graph):
                    new_states.append(modified_graph)
        
        # remove_some_edge
        for edge in graph.edges:
            modified_graph = graph.copy()
            modified_graph.remove_edge(*edge)
            if self._satisfies_constraints(modified_graph):
                new_states.append(modified_graph)
        
        # remove_add_some_edges
        for rm_edge in graph.edges:
            for add_edge in self.link_indices:
                if add_edge not in graph.edges:
                    modified_graph = graph.copy()
                    modified_graph.remove_edge(*rm_edge)
                    modified_graph.add_edge(*add_edge)
                    if self._satisfies_constraints(modified_graph) and modified_graph not in new_states:
                        new_states.append(modified_graph)
        
        return new_states

    def _satisfies_constraints(self, graph):
        for constraint in self.constraints:
            if all(edge in graph.edges for edge in constraint):
                return False
        return True

    def get_reconf_seq(self, initial_graph, target_graph):
        """
        初期グラフから目標グラフへの遷移系列を幅優先探索で求める
        """
        queue = [initial_graph]
        level_list = [0]
        
        current_graph = queue.pop(0)
        next_graphs = self.transition(current_graph)
        branches = len(next_graphs)
        level_list.append(level_list[-1] + branches)
        queue.extend(next_graphs)
        
        level = 0      
        while len(level_list) <= self.k:
            level += 1
            if level == level_list[-2]+1:
                print(f"Level {len(level_list)}: {level_list[-1]-level_list[-2]}")
                visited = list()
            current_graph = queue.pop(0)
            next_graphs = self.transition(current_graph)
            branches += len(next_graphs)            
            for next_graph in next_graphs:
                if set(next_graph.edges) not in visited:
                    queue.append(next_graph)
                    visited.append(set(next_graph.edges))
            print(level-level_list[-2], "/", level_list[-1]-level_list[-2])
            if level == level_list[-1]:
                level_list.append(level_list[-1] + len(visited))
        if set(target_graph.edges) in visited:
            print(f"Level {len(level_list)}: {level_list[-1]+branches+1}")
            return visited
        return []
            
    def make_initial_target_graphs(self):
        initial_graph = nx.DiGraph()
        target_graph = nx.DiGraph()

        for e in self.G.edges():
            # inとついている方がinノード
            u, v = e
            if (u,v) in self.initial_link_indices:
                initial_graph.add_edge(u, v)
            if (u,v) in self.target_link_indices:
                target_graph.add_edge(u, v)
        
        return initial_graph, target_graph

    def reconfiguration(self):
        self.make_constraints()
        initial_graph, target_graph = self.make_initial_target_graphs()
        reconf_seq = self.get_reconf_seq(initial_graph, target_graph)
        return reconf_seq


class ReconfigBFS_MPC(ReconfigBFS):
    def __init__(self, params, initial_graph, k, constraints_csv=None, output_path=None):
        super().__init__(params, initial_graph, initial_graph, k, constraints_csv, output_path)
        self.initial_link_indices = [(i, j) for idx, (i, j) in enumerate(self.link_indices) if initial_graph[idx] == 1]
    
    def get_reconf_seq(self, initial_graph):
        """
        初期グラフから目標グラフへの遷移系列を幅優先探索で求める
        """
        queue = [initial_graph]
        level_list = [0]
        transition_graphs = []
        
        current_graph = queue.pop(0)
        next_graphs = self.transition(current_graph)
        branches = len(next_graphs)
        level_list.append(level_list[-1] + branches)
        queue.extend(next_graphs)
        transition_graphs.append(next_graphs)
        
        level = 0      
        while len(level_list) <= self.k:
            level += 1
            if level == level_list[-2]+1:
                print(f"Level {len(level_list)}: {level_list[-1]-level_list[-2]}")
                visited = list()
            current_graph = queue.pop(0)
            next_graphs = self.transition(current_graph)
            branches += len(next_graphs)            
            for next_graph in next_graphs:
                if set(next_graph.edges) not in visited:
                    queue.append(next_graph)
                    visited.append(set(next_graph.edges))
            print(level-level_list[-2], "/", level_list[-1]-level_list[-2])
            if level == level_list[-1]:
                level_list.append(level_list[-1] + len(visited))
                transition_graphs.append(visited)
        print(f"Level {len(level_list)}: {level_list[-1]-level_list[-2]}")
        return transition_graphs
    
    def reconfiguration(self):
        self.make_constraints()
        initial_graph, _ = self.make_initial_target_graphs()
        reconf_seq = self.get_reconf_seq(initial_graph)
        return reconf_seq

class ReconfigRandomGraphZDD():
    def __init__(self, k, num_links):
        self.k = k
        self.num_links = num_links
        self.num_zones = int(math.sqrt(num_links) + 2)
        assert num_links % 2 == 0, "The number of links must be even."
        
        # num_zones個のゾーンを持つランダムなグラフを作成
        G_undirected = nx.gnm_random_graph(self.num_zones, int(self.num_links/2), directed=False)
        # 無向グラフを有向グラフに変換（各エッジの反対向きも追加される）
        G_directed = G_undirected.to_directed()

        self.adj_matrix = nx.to_numpy_array(G_directed, nodelist=range(self.num_zones))
        # リンクの総数を隣接行列に基づいて計算
        self.link_indices = [(i, j) for i in range(self.num_zones) for j in range(self.num_zones) if self.adj_matrix[i, j] == 1]
        #print("making graph...")
        self.make_graph()
        #print("done")
        #print("making constraints...")
        self.make_constraints()
        #print("done")
        
        # ランダムに初期グラフと目標グラフを作成
        self.initial_link_indices = self.generate_random_graph()
        self.target_link_indices = self.generate_random_graph()
        
    def make_graph(self):
        self.G = nx.Graph()
        self.G_edges = set()
        # 各エッジを入ノードと出ノードに分けて追加
        for u, v in self.link_indices:
            self.G.add_edge(f"{u}_out", f"{v}_in")
            self.G_edges.add((f"{u}_out", f"{v}_in"))
        # エッジリストからグラフを作成
        GraphSet.set_universe(self.G_edges)
        
    def generate_random_graph(self):
        try:
            return self.constraints.choice()
        except:
            raise ValueError("No graph satisfies the constraints.")

    def make_constraints(self):
        constraints = GraphSet({})
        # 相対するエッジが同時に存在しない
        opposite_edges = [[(f"{i}_out", f"{j}_in"), (f"{j}_out", f"{i}_in")] for i, j in self.link_indices]
        non_opposite_sets = [GraphSet({}).excluding(edges) for edges in opposite_edges]
        if non_opposite_sets:
            non_opposite_graph = reduce(operator.and_, non_opposite_sets)
        else:
            non_opposite_graph = GraphSet()  # 制約がなければ空集合
        # 制約として、全体集合から相反するグラフを除く
        constraints &= non_opposite_graph
        
        # ゾーンから出られない状態は発生しない
        all_in_edges = [[(f"{j}_out", f"{i}_in") for j in range(self.num_zones) if self.adj_matrix[j, i] == 1] for i in range(self.num_zones)]
        non_in_edge_sets = [GraphSet({}).excluding(edges) for edges in all_in_edges]
        if non_in_edge_sets:
            non_in_edges = reduce(operator.and_, non_in_edge_sets)
        else:
            non_in_edges = GraphSet()
        constraints &= non_in_edges
        self.constraints = constraints
        #print(f"Number of search space: {constraints.len()}")
        #print(f"Number of eliminated graphs: {GraphSet({}).len() - constraints.len()}")
        
    def transition(self, sz, constraints):
        return (sz.add_some_edge() | sz.remove_some_edge() | sz.remove_add_some_edges() | sz) & constraints

    def _get_reconf_set(self, setset_seq, s, t, search_space):
        '''
        終端状態から初期状態まで逆向きに辿って，遷移候補集合を取得するための関数 (目標グラフから後ろ向きに探索)
        '''
        current_set = GraphSet([t])
        reconf_set = [current_set]
        # 終端状態の一つ前のグラフ集合から初期状態まで逆向きに辿る
        for i in range(len(setset_seq) - 2, -1, -1):
            # 初期グラフから遷移可能なグラフ集合setset_seq[i]の中で，目標グラフから遷移可能なグラフ集合next_ssに含まれるグラフを選択
            next_ss = self.transition(current_set, setset_seq[i])
            if next_ss:
                current_set = next_ss
                reconf_set.append(current_set)
            else:
                print("No transition")
                return []
        return reconf_set[::-1]
    
    def get_reconf_seq(self, s, t, search_space):
        """
        目標グラフまでの遷移過程を取得する関数 (初期グラフから前向きに探索)
        """
        setset_seq = []  # 初期状態から遷移できるグラフの集合
        setset_seq.append(GraphSet([s]))
        
        while len(setset_seq) <= self.k:
            next_ad = setset_seq[-1].add_some_edge()
            next_rm = setset_seq[-1].remove_some_edge()
            next_tj = setset_seq[-1].remove_add_some_edges()
            next_ss = (next_ad | next_rm | next_tj | setset_seq[-1]) & search_space
            setset_seq.append(next_ss)
            #print(len(setset_seq), len(next_ss))
            
        if t in next_ss:
            #print("遷移系列あり")
            return self._get_reconf_set(setset_seq, s, t, search_space)
        else:
            print("遷移系列なし")
            return []
    
    def reconfiguration(self):
        """
        reconfiguration処理全体を実行する関数
        """
        reconf_seq = self.get_reconf_seq(
            self.initial_link_indices,
            self.target_link_indices,
            self.constraints
        )
        return reconf_seq
    
    
class ReconfigRandomGraphZDD_MPC(ReconfigZDD_MPC):
    def __init__(self, k, num_links):
        self.k = k
        self.num_links = num_links
        self.num_zones = int(math.sqrt(num_links) + 2) # int(num_links/4)#
        assert num_links % 2 == 0, "The number of links must be even."
        
        # num_zones個のゾーンを持つランダムなグラフを作成
        G_undirected = nx.gnm_random_graph(self.num_zones, int(self.num_links/2), directed=False)
        # 無向グラフを有向グラフに変換（各エッジの反対向きも追加される）
        G_directed = G_undirected.to_directed()

        self.adj_matrix = nx.to_numpy_array(G_directed, nodelist=range(self.num_zones))
        # リンクの総数を隣接行列に基づいて計算
        self.link_indices = [(i, j) for i in range(self.num_zones) for j in range(self.num_zones) if self.adj_matrix[i, j] == 1]
        #print("making graph...")
        self.make_graph()
        #print("done")
        #print("making constraints...")
        self.make_constraints()
        #print("done")
        
        # ランダムに初期グラフと目標グラフを作成
        #self.initial_link_indices = self.generate_random_graph()
        initial_graph = [0] * self.num_links
        self.initial_link_indices = [(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(self.link_indices) if initial_graph[idx] == 1]
        
    def make_graph(self):
        self.G = nx.Graph()
        self.G_edges = set()
        # 各エッジを入ノードと出ノードに分けて追加
        for u, v in self.link_indices:
            self.G.add_edge(f"{u}_out", f"{v}_in")
            self.G_edges.add((f"{u}_out", f"{v}_in"))
        # エッジリストからグラフを作成
        GraphSet.set_universe(self.G_edges)
        
    def generate_random_graph(self):
        try:
            return self.constraints.choice()
        except:
            raise ValueError("No graph satisfies the constraints.")

    def make_constraints(self):
        constraints = GraphSet({})
        # 相対するエッジが同時に存在しない
        opposite_edges = [[(f"{i}_out", f"{j}_in"), (f"{j}_out", f"{i}_in")] for i, j in self.link_indices]
        non_opposite_sets = [GraphSet({}).excluding(edges) for edges in opposite_edges]
        if non_opposite_sets:
            non_opposite_graph = reduce(operator.and_, non_opposite_sets)
        else:
            non_opposite_graph = GraphSet()  # 制約がなければ空集合
        # 制約として、全体集合から相反するグラフを除く
        constraints &= non_opposite_graph
        
        # ゾーンから出られない状態は発生しない
        all_in_edges = [[(f"{j}_out", f"{i}_in") for j in range(self.num_zones) if self.adj_matrix[j, i] == 1] for i in range(self.num_zones)]
        non_in_edge_sets = [GraphSet({}).excluding(edges) for edges in all_in_edges]
        if non_in_edge_sets:
            non_in_edges = reduce(operator.and_, non_in_edge_sets)
        else:
            non_in_edges = GraphSet()
        constraints &= non_in_edges
        self.constraints = constraints
        #print(f"Number of search space: {constraints.len()}")
        #print(f"Number of eliminated graphs: {GraphSet({}).len() - constraints.len()}")
        


@ray.remote(num_cpus=1)
def task_with_timeout(k, num_links):
    df = {"k": [], "num_links": [], "elapsed_time": [], "max_memory_gb": [], "memo": []}
    gc.collect()
    reconf = ReconfigRandomGraphZDD(k, num_links)
    stop_event = threading.Event()
    memory_log = []
    def monitor_memory():
        process = psutil.Process(os.getpid())
        while not stop_event.is_set():
            try:
                memory_info = process.memory_info().rss / (1024 ** 3)  # メモリ使用量 (GB)
                memory_log.append(memory_info)  # ログに追加
            except psutil.NoSuchProcess:
                break
            if num_links < 40:
                time.sleep(0.01)
            else:
                time.sleep(1)

    # メモリ使用量のモニタリングスレッドを開始
    monitor_thread = threading.Thread(target=monitor_memory,daemon=True)
    monitor_thread.start()
    start_time = time.time()
    
    try:
        _ = reconf.reconfiguration()
        elapsed_time = time.time() - start_time
        max_memory_usage = max(memory_log) if memory_log else None
        print(f"Elapsed time: {elapsed_time:.2f}s, Max memory: {max_memory_usage:.2f}GB")
        df["k"].append(k)
        df["num_links"].append(num_links)
        df["elapsed_time"].append(elapsed_time)
        df["max_memory_gb"].append(max_memory_usage)
        df["memo"].append("Success")
    except Exception as e:
        df["k"].append(k)
        df["num_links"].append(num_links)
        df["elapsed_time"].append(None)
        df["max_memory_gb"].append(None)
        df["memo"].append(str(e))
    finally:
        stop_event.set()
        monitor_thread.join()
    return df
        
def calc_performance(k_list, num_links_list, timeout_seconds, num_cpus):
    results = []
    for num_links in num_links_list:
        for k in k_list:
            print(f"Number of links: {num_links}, k: {k}")
            tasks = [task_with_timeout.remote(k, num_links) for _ in range(10)]
            gc.collect()
            finished, tasks = ray.wait(tasks, num_returns=len(tasks), timeout=timeout_seconds)
            print(f"Finished {len(finished)} tasks")
            if len(finished) < len(tasks):
                for _ in range(len(tasks) - len(finished)):
                    results.append([{ "k": k, "num_links": num_links, "elapsed_time": None, "max_memory_gb": None, "memo": "Timeout" }])
            else:
                results.extend(ray.get(finished))
            gc.collect()
            
            #メモリオーバーフローを防ぐため一定時間後にrayプロセスを終了
            ray.shutdown()
            ray.init(num_cpus=num_cpus)
                
    # 結果の保存
    final_results = pd.concat([pd.DataFrame(r) for r in results])
    final_results.to_csv("../../output/reconfiguration/reconf_performance.csv", index=False)
    
    # メモリ使用量と計算時間の可視化
    fig, ax = plt.subplots(figsize=(10, 6))
    df = final_results.groupby(["k", "num_links"])[["elapsed_time", "max_memory_gb"]].mean().reset_index()
    for k in k_list:
        df_k = df[df["k"] == k]
        ax.plot(df_k["num_links"], df_k["elapsed_time"], label=f"k={k}")
    ax.set_xlabel("Number of links")
    ax.set_ylabel("Elapsed time (s)")
    ax.set_title("Elapsed time")
    ax.legend()
    plt.savefig("../../output/reconfiguration/elapsed_time.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in k_list:
        df_k = df[df["k"] == k]
        ax.plot(df_k["num_links"], df_k["max_memory_gb"], label=f"k={k}")
    ax.set_xlabel("Number of links")
    ax.set_ylabel("Max memory usage (GB)")
    ax.set_title("Max memory usage")
    ax.legend()
    plt.savefig("../../output/reconfiguration/max_memory_usage.png")
    plt.close()
        
        
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    
    initial_graph = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    target_graph =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    #target_graph =  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    k = 2
    print(f"Hammimg distance: {sum([1 for i, j in zip(initial_graph, target_graph) if i != j])}")
    
    params = Parameters()
    # constraints_csv = pd.read_csv("../../output/constraint_all/constraints.csv")
    constraints_csv = None
    
    # ZDD
    #reconf = ReconfigZDD(params, initial_graph, target_graph, k=k, constraints_csv=constraints_csv, output_path=None)
    reconf = ReconfigZDD_MPC(params, initial_graph, k=k, constraints_csv=constraints_csv, output_path=None)
    # BFS
    #reconf = ReconfigBFS(params, initial_graph, target_graph, k=k, constraints_csv=constraints_csv, output_path=None)
    #reconf = ReconfigBFS_MPC(params, initial_graph, k=k, constraints_csv=constraints_csv, output_path=None)
    # A star
    #reconf = ReconfigAstar(params, initial_graph, target_graph, constraints_csv, output_path="../../output/reconfiguration")
    # DFS
    #reconf = ReconfigDFS(params, initial_graph, target_graph, constraints_csv, output_path="../../output/reconfiguration")
    
    reconf_seq = reconf.reconfiguration()
    print(len(reconf_seq[-1]))
    
    '''
    k_list = [4]#, 20]
    num_links_list = [20]
    calc_performance(k_list, num_links_list, timeout_seconds=10*60, num_cpus=10)
    '''