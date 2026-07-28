"""
===============================================================================
ECHO-BASIN GRAPH ANOMALY DETECTION (GAD) - HUB FINDER & HOMOPHILY BOUNDARY PIPELINE
===============================================================================

1. What is Barabási–Albert (BA) Scale-Free Network Theory?
----------------------------------------------------------
In 1999, Albert-László Barabási and Réka Albert discovered that real-world networks 
(social networks, financial transaction graphs, the internet) do not have a 
random distribution of connections.

Instead, they grow via Preferential Attachment ("the rich get richer"):
When new users/nodes join a network, they preferentially connect to nodes 
that are already well-connected.

This creates a Power-Law Degree Distribution: 
    P(k) ~ k^(-gamma)   (typically gamma approx 2.5 - 3.0)

The Key Insight: In scale-free graphs, a tiny fraction of nodes naturally 
become structural hubs, while the vast majority of nodes are peripheral.


2. Step-by-Step Derivation of K_ideal = ceil( log2(N) * sqrt(k_max) )
----------------------------------------------------------------------

Step 1: Maximum Degree & Core Shell Bounds
In a scale-free graph of N nodes with power-law exponent gamma = 3:
The maximum node degree k_max_deg scales with network size as: 
    k_max_deg ~ N^(1 / (gamma - 1)) = N^(1/2) = sqrt(N)

Dorogovtsev et al. ("K-core architecture of complex networks", 2006) proved that 
the maximum k-coreness level (k_max) of a scale-free network is bounded by: 
    k_max ~ sqrt(k_max_deg) ~ N^(1/4)

This tells us that k_max measures the density depth of the network.


Step 3: Combining Diameter Scaling with Core Depth
To find the ideal hub count K:
- We need log2(N) base hubs to cover the N-scale logarithmic diameter of the network.
- We scale this base by the Core Depth Factor sqrt(k_max) to account for 
  multi-tier community density (k_max).

Multiplying these two topological properties together gives:
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
    Computes node coreness. Uses NetworkX for N <= 50,000 nodes, 
    and fast PyTorch/DGL vector operations for massive graphs.
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
    Derives ideal hub count K adaptively using Barabási-Albert Scale-Free Network Theory:
    K_ideal = ceil(log2(N) * sqrt(k_max))
    """
    coreness = compute_coreness_fast(g)
    num_nodes = g.num_nodes()
    max_k = max(int(coreness.max().item()), 1)
    
    k_ideal = int(math.ceil(math.log2(max(num_nodes, 2)) * math.sqrt(max_k)))
    return max(k_ideal, 1)

def get_kcore_hubs(g, top_k=None):
    """
    K-Core Hub Finder for Graph Anomaly Detection.
    """
    coreness = compute_coreness_fast(g)
    norm_coreness = coreness / (coreness.max() + 1e-6)
    
    if top_k is None:
        top_k = derive_k_ideal(g)
        
    top_scores, top_hubs = torch.topk(norm_coreness, min(top_k, g.num_nodes()))
    return top_hubs, top_scores, norm_coreness

def _get_ego_neighbors(g, hub, m_hops=2):
    """Extracts 1-hop and 2-hop neighbor nodes for a given hub, ensuring idtype compatibility."""
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

def build_homophily_boundaries(g, feats, hub_indices, tau=0.60, m_hops=2):
    """
    Stage 1 Part B: Homophily Boundary Construction around Hubs.
    Formulation: Fixed m-Hop Cosine Thresholding (tau=0.60).
    """
    feats_norm = F.normalize(feats, p=2, dim=-1)
    chambers = {}
    all_chamber_nodes = set()
    
    for hub in hub_indices.tolist():
        candidate_nodes = _get_ego_neighbors(g, hub, m_hops=m_hops)
        hub_feat = feats_norm[hub]
        cand_feats = feats_norm[candidate_nodes]
        
        sims = (cand_feats * hub_feat).sum(dim=-1)
        valid_mask = sims >= tau
        
        chamber_nodes = candidate_nodes[valid_mask].tolist()
        chambers[hub] = chamber_nodes
        all_chamber_nodes.update(chamber_nodes)
        
    num_nodes = g.num_nodes()
    isolates = list(set(range(num_nodes)) - all_chamber_nodes)
    
    stats = {
        'num_hubs': len(hub_indices),
        'num_chambers': len(chambers),
        'num_in_chamber_nodes': len(all_chamber_nodes),
        'num_isolate_nodes': len(isolates),
        'chamber_coverage_ratio': len(all_chamber_nodes) / num_nodes,
        'isolate_ratio': len(isolates) / num_nodes
    }
    
    return chambers, isolates, stats
