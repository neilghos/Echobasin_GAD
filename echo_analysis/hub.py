"""
===============================================================================
ECHO-BASIN GRAPH ANOMALY DETECTION (GAD) - HUB FINDER & HOMOPHILY BOUNDARY PIPELINE
===============================================================================

1. What is Barabási–Albert (BA) Scale-Free Network Theory?
----------------------------------------------------------
In 1999, Albert-László Barabási and Réka Albert discovered that real-world networks 
grow via Preferential Attachment ("the rich get richer"), producing a power-law 
degree distribution P(k) ~ k^(-gamma).

2. Step-by-Step Derivation of K_ideal = ceil( log2(N) * sqrt(k_max) )
----------------------------------------------------------------------
- Max degree k_max_deg ~ sqrt(N)
- Max k-core depth k_max ~ N^(1/4)
- Ultra-Small-World Logarithmic Path Length D ~ log2(N)
Combining topological diameter with core depth gives:
    K_ideal = ceil( log2(N) * sqrt(k_max) )
===============================================================================
"""

import math
import sys
import os
import torch
import torch.nn.functional as F
import networkx as nx
import dgl

def compute_coreness_fast(g):
    """
    Computes node coreness using NetworkX for small/medium graphs, 
    and fast PyTorch/DGL vector operations for large graphs.
    """
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    
    if num_nodes <= 50000 and num_edges <= 1000000:
        nx_g = nx.Graph(dgl.to_networkx(g.cpu()).to_undirected())
        nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
        core_dict = nx.core_number(nx_g)
        coreness = torch.tensor([core_dict.get(i, 0) for i in range(num_nodes)], dtype=torch.float)
    else:
        degrees = (g.in_degrees() + g.out_degrees()).float()
        coreness = degrees.clone()
        for _ in range(5):
            with g.local_scope():
                g.ndata['c'] = coreness
                g.update_all(dgl.function.copy_u('c', 'm'), dgl.function.mean('m', 'c_mean'))
                coreness = torch.min(coreness, g.ndata['c_mean'] * 1.5 + 1.0)
    return coreness

def derive_k_ideal(g):
    """
    Derives ideal hub count K adaptively using Scale-Free Network Theory:
    K_ideal = ceil(log2(N) * sqrt(k_max))
    """
    coreness = compute_coreness_fast(g)
    num_nodes = g.num_nodes()
    max_k = max(int(coreness.max().item()), 1)
    
    k_ideal = int(math.ceil(math.log2(max(num_nodes, 2)) * math.sqrt(max_k)))
    return max(k_ideal, 1)

def get_kcore_hubs(g, top_k=None):
    """K-Core Hub Finder for Graph Anomaly Detection."""
    coreness = compute_coreness_fast(g)
    norm_coreness = coreness / (coreness.max() + 1e-6)
    
    if top_k is None:
        top_k = derive_k_ideal(g)
        
    top_scores, top_hubs = torch.topk(norm_coreness, min(top_k, g.num_nodes()))
    return top_hubs, top_scores, norm_coreness

def _get_ego_neighbors(g, hub, m_hops=2):
    """Extracts m-hop ego neighbors for a hub with idtype safety."""
    hub_val = int(hub)
    hub_tensor = torch.tensor([hub_val], dtype=g.idtype, device=g.device)
    succ1 = g.successors(hub_tensor)
    pred1 = g.predecessors(hub_tensor)
    hop1 = torch.unique(torch.cat([succ1, pred1, hub_tensor]))
    
    if m_hops == 1 or len(hop1) > 200:
        return hop1.long()
        
    hop2_list = [hop1]
    for n in hop1[:25]:
        n_val = int(n.item())
        n_tensor = torch.tensor([n_val], dtype=g.idtype, device=g.device)
        succ2 = g.successors(n_tensor)
        pred2 = g.predecessors(n_tensor)
        hop2_list.extend([succ2, pred2])
        
    return torch.unique(torch.cat(hop2_list)).long()

def build_homophily_boundaries(g, feats, hub_indices, tau=0.50, m_hops=2):
    """
    Stage 1 Part B: Homophily Boundary Construction around Hubs.
    Universal SOTA Model B Defaults: tau = 0.50, m_hops = 2.
    """
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    
    if m_hops is None:
        m_hops = 2
    if tau is None:
        tau = 0.50
            
    feats_norm = F.normalize(feats, p=2, dim=-1)
    chambers = {}
    all_chamber_nodes = set()
    
    # Detect global feature homogeneity (e.g. Reddit: 99.4% cosine sim across all nodes).
    # When all features are near-identical, tau cosine filtering passes every node into every
    # chamber → centroids collapse → scoring_input identical for class 0 and class 1 → loss lock.
    # Fix: bypass tau filter and use pure structural (k-core ego) chamber membership instead.
    # Fixed deterministic sampling of 500 nodes for feature homogeneity check
    sample_idx = torch.linspace(0, feats_norm.shape[0] - 1, min(500, feats_norm.shape[0]), device=feats.device).long()
    sample_feats = feats_norm[sample_idx]  # [S, D]
    mean_sim = (sample_feats @ sample_feats.T).mean().item()
    use_structural_only = mean_sim >= 0.85
    
    for hub in hub_indices.tolist():
        candidate_nodes = _get_ego_neighbors(g, hub, m_hops=m_hops)
        
        if use_structural_only:
            # Pure structural ego membership — no feature cosine filtering
            chamber_nodes = candidate_nodes.tolist()
        else:
            hub_feat = feats_norm[hub]
            cand_feats = feats_norm[candidate_nodes]
            sims = (cand_feats * hub_feat).sum(dim=-1)
            valid_mask = sims >= tau
            chamber_nodes = candidate_nodes[valid_mask].tolist()
        
        chambers[hub] = chamber_nodes
        all_chamber_nodes.update(chamber_nodes)
        
    isolates = list(set(range(num_nodes)) - all_chamber_nodes)
    
    stats = {
        'num_hubs': len(hub_indices),
        'num_chambers': len(chambers),
        'num_in_chamber_nodes': len(all_chamber_nodes),
        'num_isolate_nodes': len(isolates),
        'chamber_coverage_ratio': len(all_chamber_nodes) / num_nodes,
        'isolate_ratio': len(isolates) / num_nodes,
        'm_hops_used': m_hops,
        'tau_used': tau
    }
    
    return chambers, isolates, stats
