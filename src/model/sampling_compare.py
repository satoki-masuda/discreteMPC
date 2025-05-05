import numpy as np
import pandas as pd
import networkx as nx
import time
import random
import math
from graphillion import setset, GraphSet
from parameters_ndp import Parameters
from reconf_horizon import ReconfigZDD_MPC, ReconfigRandomGraphZDD_MPC

def array_to_zdd(array, link_indices):
    """
    1, 2次元配列 (row: graph数, col: リンクの0/1)をZDDに変換する関数。
    """
    if len(array.shape) == 1:
        gs = [[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(link_indices) if array[idx] == 1]]
    else:
        gs = [[(f"{i}_out", f"{j}_in") for idx, (i, j) in enumerate(link_indices) if array[row, idx] == 1] for row in range(array.shape[0])]
    return GraphSet(gs)

def zdd_to_array(zdd, link_indices):
    """
    ZDDを2次元配列 (row: graph数, col: リンクの0/1)に変換する関数。
    """
    array = [[1 if (f"{i}_out", f"{j}_in") in g or (f"{j}_in", f"{i}_out") in g else 0 for i, j in link_indices] for g in list(zdd)]
    return array

def make_graph_from_index(link_config, link_indices, G_dash):
    # 入ノードと出ノードに分けずにエッジを追加
    target_links = [(i, j) for idx, (i, j) in enumerate(link_indices) if link_config[idx] == 1]
    graph = nx.DiGraph()
    for e in G_dash.edges():
        u, v = e
        if (u,v) in target_links:
            graph.add_edge(u, v)
    return graph

def transition(graph, link_indices, constraints):
    new_states = [graph]
    # add_some_edge
    for edge in link_indices:
        if edge not in graph.edges:
            modified_graph = graph.copy()
            modified_graph.add_edge(*edge)
            if satisfies_constraints(modified_graph, constraints):
                new_states.append(modified_graph)
    
    # remove_some_edge
    for edge in graph.edges:
        modified_graph = graph.copy()
        modified_graph.remove_edge(*edge)
        if satisfies_constraints(modified_graph, constraints):
            new_states.append(modified_graph)
    
    # remove_add_some_edges
    for rm_edge in graph.edges:
        for add_edge in link_indices:
            if add_edge not in graph.edges:
                modified_graph = graph.copy()
                modified_graph.remove_edge(*rm_edge)
                modified_graph.add_edge(*add_edge)
                if satisfies_constraints(modified_graph, constraints) and modified_graph not in new_states:
                    new_states.append(modified_graph)
    
    return new_states

def satisfies_constraints(graph, constraints):
    for constraint in constraints:
        if all(edge in graph.edges for edge in constraint):
            return False
    return True

num_link_list = [10, 20, 30, 40, 50, 60]
k = 4
repititions = 5
df = pd.DataFrame(np.zeros((len(num_link_list), 4)),
                  columns=["num_links", "t_const", "t_zdd", "t_naive"])
df["num_links"] = num_link_list

for num_links in num_link_list:
    
    t_const = 0
    t_zdd = 0
    t_naive = 0
    for rep in range(repititions):
        print(f"num_links: {num_links}, rep: {rep+1}")
        reconf = ReconfigRandomGraphZDD_MPC(k=k, num_links=num_links)
        num_zones = reconf.num_zones
        link_indices = [(i, j) for i in range(num_zones) for j in range(num_zones) if reconf.adj_matrix[i, j] == 1]
        population_size = 1000
        noise_scale = 1/40 # 毎秒の状態変数xの摂動のスケール．noise_scale*xが摂動の標準偏差．
        control_horizon = k * 60 # min. 遷移のホライゾン
        prediction_horizon = k * 60 # min. 遷移のホライゾン
        mpc_start_time = 12 * 60
        mpc_end_time = 18 * 60
        seq_interval = 60 # min. 遷移のステップ幅
        time_steps = int(control_horizon // seq_interval)
        dim = num_links * (time_steps)
        p = np.full(dim, 0.1)
        
        ##### ZDD #####
        start_time = time.time()
        zdd_constraints = reconf.reconfiguration()
        t = time.time() - start_time
        print(f"t_const = {t}")
        t_const += t
        

        population = np.zeros((population_size, num_links*(time_steps)))
        costs = []
        gen_times = []
        concat_list = []
        n_sample = 0
        # 1段階目のサンプリング：各候補は dim 次元のバイナリベクトル
        print("Sampling initial population...")
        start_time = time.time()
        count = 0
        while n_sample < population_size:
            # ランダムな制御変数を生成
            tmp = (np.random.rand(100, num_links) < p[:num_links]).astype(int)
            # ZDDの制約条件を満たすかどうかを確認
            tmp = array_to_zdd(tmp, link_indices) & zdd_constraints[1]
            if len(tmp) > 0:
                concat_list.extend(zdd_to_array(tmp, link_indices))
                n_sample += len(tmp)
            count += 1
        population[:,:num_links] = np.concatenate(np.array(concat_list)[:population_size], axis=0).reshape(population_size, num_links)
        gen_times.append(time.time() - start_time)

        invalid_idx = []
        for t in range(1, time_steps):
            print(f"Sampling population for time step {t}...")
            start_time = time.time()
            # 同じ個体は一気に処理する
            group, indices = np.unique(population[:,((t-1)*num_links):(t*num_links)], axis=0, return_inverse=True)
            for idx, ind in enumerate(group):
                arr = []
                n_sample = 0
                target_count = np.sum(indices == idx)
                count = 0
                while n_sample < target_count:
                    if count > 1000:
                        print("Too many iterations and forced to break")
                        #arr.append(np.array([np.zeros_like(ind) for _ in range(target_count - n_sample)]))
                        break
                    # ランダムな制御変数を生成
                    tmp = (np.random.rand(10**2, num_links) < p[(t*num_links):((t+1)*num_links)]).astype(int)
                    # ZDDの制約条件を満たす、かつ、現在状態から遷移可能なもののみを選択
                    zdd_tmp = array_to_zdd(tmp, link_indices) & zdd_constraints[t+1] & reconf.transition(array_to_zdd(ind, link_indices), reconf.constraints)
                    n = min(len(zdd_tmp), target_count - n_sample)
                    if n > 0:
                        arr.extend(zdd_to_array(zdd_tmp, link_indices)[:n])
                        n_sample += n
                    count += 1
                if n_sample == target_count:
                    population[indices == idx, (t*num_links):((t+1)*num_links)] = np.concatenate(np.array(arr), axis=0).reshape(target_count, num_links)
                elif n_sample > 0:
                    # arrの長さがtarget_countより小さい場合、残りの部分はarrの最後の要素で埋める
                    arr.extend([arr[-1] for _ in range(target_count - n_sample)])
                    population[indices == idx, (t*num_links):((t+1)*num_links)] = np.concatenate(np.array(arr), axis=0).reshape(target_count, num_links)
                else:
                    # arrが空の場合、idxを記録
                    invalid_idx.append(idx)
            gen_times.append(time.time() - start_time)
        # 無効なインデックスを持つ個体は、有効な要素の個体のコピーで置換
        if len(invalid_idx) > 0:
            valid_indices = np.setdiff1d(np.arange(population_size), invalid_idx)
            for idx in invalid_idx:
                population[idx] = population[random.choice(valid_indices)].copy()

        t_zdd += sum(gen_times)
        print(f"gen_times: {gen_times}, sum: {sum(gen_times)}")


        #### naive ####
        constraints = []
        # 相対するエッジが同時に存在しない
        opposite_edges = [[(i, j), (j, i)] for i, j in link_indices]
        constraints.extend(opposite_edges)
        # ゾーンから出られない状態は発生しない
        all_in_edges = [[(j, i) for j in range(num_zones) if reconf.adj_matrix[j, i] == 1] for i in range(num_zones)]
        constraints.extend(all_in_edges)

        G_dash = nx.DiGraph()
        for u, v in link_indices:
            G_dash.add_edge(u, v)
        # 入ノードと出ノードに分けずにエッジを追加
        initial_link = np.zeros(num_links)
        initial_graph = make_graph_from_index(initial_link, link_indices, G_dash)

        population = np.zeros((population_size, num_links*(time_steps)))
        costs = []
        gen_times = []
        concat_list = []
        n_sample = 0
        # 1段階目のサンプリング：各候補は dim 次元のバイナリベクトル
        print("Sampling initial population...")
        start_time = time.time()
        new_states = transition(initial_graph, link_indices, constraints)
        while n_sample < population_size:
            # ランダムな制御変数を生成
            tmp = (np.random.rand(num_links) < p[:num_links]).astype(int)
            graph = make_graph_from_index(tmp, link_indices, G_dash)
            # ZDDの制約条件を満たすかどうかを確認
            if satisfies_constraints(graph, constraints) and set(graph.edges()) in [set(cand.edges()) for cand in new_states]:
                concat_list.append(tmp)
                n_sample += 1
        population[:,:num_links] = np.concatenate(np.array(concat_list)[:population_size], axis=0).reshape(population_size, num_links)
        gen_times.append(time.time() - start_time)

        invalid_idx = []
        for t in range(1, time_steps):
            print(f"Sampling population for time step {t}...")
            start_time = time.time()
            # 同じ個体は一気に処理する
            group, indices = np.unique(population[:,((t-1)*num_links):(t*num_links)], axis=0, return_inverse=True)
            for idx, ind in enumerate(group):
                arr = []
                n_sample = 0
                target_count = np.sum(indices == idx)
                count = 0
                ind_graph = make_graph_from_index(ind, link_indices, G_dash)
                new_states = transition(ind_graph, link_indices, constraints)
                while n_sample < target_count:
                    if count > 1000:
                        print("Too many iterations and forced to break")
                        break
                    # ランダムな制御変数を生成
                    tmp = (np.random.rand(num_links) < p[(t*num_links):((t+1)*num_links)]).astype(int)
                    graph = make_graph_from_index(tmp, link_indices, G_dash)
                    if satisfies_constraints(graph, constraints) and set(graph.edges()) in [set(cand.edges()) for cand in new_states]:
                        arr.append(tmp)
                        n_sample += 1
                        
                if n_sample == target_count:
                    population[indices == idx, (t*num_links):((t+1)*num_links)] = np.concatenate(np.array(arr), axis=0).reshape(target_count, num_links)
                elif n_sample > 0:
                    # arrの長さがtarget_countより小さい場合、残りの部分はarrの最後の要素で埋める
                    arr.extend([arr[-1] for _ in range(target_count - n_sample)])
                    population[indices == idx, (t*num_links):((t+1)*num_links)] = np.concatenate(np.array(arr), axis=0).reshape(target_count, num_links)
                else:
                    # arrが空の場合、idxを記録
                    invalid_idx.append(idx)
            gen_times.append(time.time() - start_time)
        # 無効なインデックスを持つ個体は、有効な要素の個体のコピーで置換
        if len(invalid_idx) > 0:
            valid_indices = np.setdiff1d(np.arange(population_size), invalid_idx)
            for idx in invalid_idx:
                population[idx] = population[random.choice(valid_indices)].copy()

        t_naive += sum(gen_times)
        print(f"gen_times: {gen_times}, sum: {sum(gen_times)}")
    
    df.loc[df["num_links"] == num_links, "t_const"] = t_const/repititions
    df.loc[df["num_links"] == num_links, "t_zdd"] = t_zdd/repititions
    df.loc[df["num_links"] == num_links, "t_naive"] = t_naive/repititions
    
df.to_csv("../../output/mpc/compare_sampling.csv", index=False)