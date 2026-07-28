"""
===============================================================================
ECHO-BASIN FULL DATASET BOTTLENECK DIAGNOSTIC SCRIPT (6 MAIN BENCHMARKS)
===============================================================================
Analyzes structural homophily, chamber partitioning bottlenecks, anomaly recall, 
and representation separation across Weibo, Reddit, Amazon, Yelp, Tolokers, Questions.
"""

import sys, os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

sys.path.append('/home/utsab/Data/EchoBasin-GAD/echo_analysis')

from dataloader import EchoDataLoader
from hub import get_kcore_hubs, build_homophily_boundaries, derive_k_ideal
from models import EchoBasinGADModel

ALL_DATASETS = ['weibo', 'reddit', 'amazon', 'yelp', 'tolokers', 'questions']

def analyze_dataset(dname):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loader = EchoDataLoader(dname)
    splits = loader.get_official_splits(trial_id=0)
    g = splits['g']
    feats = splits['feats']
    labels = splits['node_labels']
    
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    avg_degree = (2.0 * num_edges) / max(num_nodes, 1)
    
    # 1. Structural & Feature Homophily (sampled 10k edges to prevent OOM)
    src, dst = g.edges()
    feat_norm = F.normalize(feats, p=2, dim=-1)
    
    sample_size = min(10000, num_edges)
    perm = torch.randperm(num_edges)[:sample_size]
    s_samp, d_samp = src[perm], dst[perm]
        
    edge_cossim = (feat_norm[s_samp] * feat_norm[d_samp]).sum(dim=-1).mean().item()
    same_label_edges = (labels[s_samp] == labels[d_samp]).float().mean().item()
    
    # 2. Stage 1 Chamber Partitioning & Anomaly Recall
    k_ideal = derive_k_ideal(g)
    hubs, _, _ = get_kcore_hubs(g, top_k=k_ideal)
    chambers, isolates, stats = build_homophily_boundaries(g, feats, hubs)
    
    all_anomalies = set((labels == 1).nonzero(as_tuple=True)[0].tolist())
    total_anomalies = len(all_anomalies)
    
    chamber_nodes = set().union(*chambers.values()) if len(chambers) > 0 else set()
    anomalies_in_chambers = len(all_anomalies.intersection(chamber_nodes))
    anomalies_in_isolates = len(all_anomalies.intersection(set(isolates)))
    
    anomaly_recall_in_isolates = anomalies_in_isolates / max(total_anomalies, 1)
    
    # 3. Model Forward Pass & Representation Separation
    model = EchoBasinGADModel(in_dim=loader.in_dim, hidden_dim=64, out_dim=64).to(device)
    edge_index = torch.stack([src[perm], dst[perm]], dim=0).to(device)
        
    model.eval()
    with torch.no_grad():
        prob_anomaly, h = model(feats.to(device), edge_index, chambers, isolates)
        
    h_norm = F.normalize(h, p=2, dim=-1)
    if len(chambers) > 0:
        hub_list = list(chambers.keys())[:100]
        centroids = torch.stack([h[chambers[hub]].mean(dim=0) for hub in hub_list], dim=0)
        c_norm = F.normalize(centroids, p=2, dim=-1)
        affinity = torch.matmul(h_norm, c_norm.T).clamp(0, 1).max(dim=-1).values
        
        anom_mask = (labels == 1)
        norm_mask = (labels == 0)
        
        aff_anom = affinity[anom_mask].mean().item()
        aff_norm = affinity[norm_mask].mean().item()
        aff_gap = aff_norm - aff_anom
    else:
        aff_anom, aff_norm, aff_gap = 0.0, 0.0, 0.0
        
    return {
        'Dataset': dname,
        'Nodes': num_nodes,
        'Edges': num_edges,
        'Avg Deg': f"{avg_degree:.1f}",
        'Edge CosSim': f"{edge_cossim:.3f}",
        'Label Homo': f"{same_label_edges:.3f}",
        'Hubs K': k_ideal,
        'Chamber Nodes': len(chamber_nodes),
        'Isolate Nodes': len(isolates),
        'Total Anom': total_anomalies,
        'Anom in Chm': anomalies_in_chambers,
        'Anom in Iso': anomalies_in_isolates,
        'Iso Anom Recall': f"{anomaly_recall_in_isolates*100:.1f}%",
        'Affinity Gap': f"{aff_gap:.3f}"
    }

def main():
    print("\n==========================================================================")
    print("      ECHO-BASIN FULL DATASET BOTTLENECK DIAGNOSTIC SUITE")
    print("==========================================================================\n")
    
    results = []
    for dname in ALL_DATASETS:
        res = analyze_dataset(dname)
        results.append(res)
        
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("\n==========================================================================\n")

if __name__ == '__main__':
    main()
