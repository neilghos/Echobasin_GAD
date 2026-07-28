import sys
import os
import torch
import numpy as np
import networkx as nx
import dgl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import Dataset
from hub import get_kcore_hubs

def derive_adaptive_hub_count(g, method='max_core'):
    """
    Derives the ideal number of hubs K adaptively from graph topology.
    
    Methods:
    - 'max_core': Selects all nodes in the top 10% max k-core shells (k >= max_k - 1)
    - 'power_law_elbow': Finds the elbow / knee point on the sorted coreness distribution curve.
    - 'fractional_log': K = ceil(log2(N) * sqrt(max_k))
    """
    nx_g = nx.Graph(dgl.to_networkx(g.cpu()).to_undirected())
    nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
    
    core_dict = nx.core_number(nx_g)
    core_vals = np.array(list(core_dict.values()))
    max_k = core_vals.max()
    num_nodes = g.num_nodes()
    
    if method == 'max_core':
        # Select all nodes in highest and 2nd highest k-shells
        threshold_k = max(1, max_k - 1)
        hubs = np.where(core_vals >= threshold_k)[0]
        k_count = len(hubs)
        
    elif method == 'power_law_elbow':
        # Sort coreness descending
        sorted_cores = np.sort(core_vals)[::-1]
        # Find maximum drop / curvature point
        diffs = np.diff(sorted_cores)
        elbow_idx = np.argmin(diffs) + 1 if len(diffs) > 0 else 10
        k_count = max(5, int(elbow_idx))
        
    elif method == 'fractional_log':
        # Scale-free topological heuristic
        k_count = int(np.ceil(np.log2(num_nodes) * np.sqrt(max_k)))
        
    return k_count, max_k

def analyze_hub_counts():
    print("Loading Weibo dataset...")
    dataset = Dataset('weibo-els', prefix='/home/utsab/Data/EchoBasin-GAD/datasets/edge_labels/', sp_type='star+norm', labels_have='ne')
    dataset.prepare_dataset(total_trials=1)
    g = dataset.graph_list[0]
    num_nodes = g.num_nodes()
    
    print(f"\n--- Weibo Dataset (Total Nodes: {num_nodes}) ---")
    
    k_maxcore, max_k = derive_adaptive_hub_count(g, method='max_core')
    k_elbow, _ = derive_adaptive_hub_count(g, method='power_law_elbow')
    k_log, _ = derive_adaptive_hub_count(g, method='fractional_log')
    
    print(f"Max K-Core Shell (max_k): {max_k}")
    print(f"1. Max-Core Shell Method (k >= max_k - 1): Derived K = {k_maxcore} hubs ({k_maxcore/num_nodes:.2%} of nodes)")
    print(f"2. Power-Law Elbow Method: Derived K = {k_elbow} hubs ({k_elbow/num_nodes:.2%} of nodes)")
    print(f"3. Scale-Free Log Topological Method: Derived K = {k_log} hubs ({k_log/num_nodes:.2%} of nodes)")

if __name__ == '__main__':
    analyze_hub_counts()
