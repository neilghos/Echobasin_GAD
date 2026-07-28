import sys
import os
import torch
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import Dataset
from hub import get_kcore_hubs
from chamber_builder import HomophilyChamberBuilder

def evaluate_chamber_heuristics():
    print("==========================================================================")
    print("   STAGE 1 PART B: HOMOPHILY CHAMBER BOUNDARY HEURISTIC ANALYSIS")
    print("==========================================================================")
    
    print("Loading Weibo Dataset...")
    dataset = Dataset('weibo-els', prefix='/home/utsab/Data/EchoBasin-GAD/datasets/edge_labels/', sp_type='star+norm', labels_have='ne')
    dataset.prepare_dataset(total_trials=1)
    g = dataset.graph_list[0]
    feats = g.ndata['feature']
    labels = dataset.node_label[0]
    
    num_nodes = g.num_nodes()
    total_anomalies = labels.sum().item()
    global_anomaly_rate = total_anomalies / num_nodes
    
    # 1. Get Top Hubs via K-Core Scale-Free Derivation
    hubs, scores, _ = get_kcore_hubs(g)
    print(f"Top K-Core Hubs Selected: {len(hubs)} hubs ({len(hubs)/num_nodes:.2%})")
    
    builder = HomophilyChamberBuilder(m_hops=2, ppr_alpha=0.15)
    
    heuristics = [
        ('Fixed m-Hop Cosine (tau=0.60)', lambda: builder.build_chambers_mhop_cosine(g, feats, hubs, tau=0.60)),
        ('Fixed m-Hop Cosine (tau=0.75)', lambda: builder.build_chambers_mhop_cosine(g, feats, hubs, tau=0.75)),
        ('PPR Random Walk Diffusion (tau=0.005)', lambda: builder.build_chambers_ppr_gate(g, feats, hubs, ppr_tau=0.005)),
        ('Adaptive Chamber Variance (alpha=0.5)', lambda: builder.build_chambers_adaptive_variance(g, feats, hubs, alpha_std=0.5))
    ]
    
    results = []
    
    for name, fn in heuristics:
        chambers, isolates = fn()
        
        # Calculate in-chamber nodes vs isolate nodes
        in_chamber_nodes = set()
        for c_nodes in chambers.values():
            in_chamber_nodes.update(c_nodes)
            
        num_in_chamber = len(in_chamber_nodes)
        num_isolates = len(isolates)
        
        # Anomaly counts in each pool
        chamber_labels = labels[list(in_chamber_nodes)] if len(in_chamber_nodes) > 0 else torch.tensor([])
        isolate_labels = labels[isolates] if len(isolates) > 0 else torch.tensor([])
        
        chamber_anom_count = chamber_labels.sum().item() if len(chamber_labels) > 0 else 0
        isolate_anom_count = isolate_labels.sum().item() if len(isolate_labels) > 0 else 0
        
        chamber_cleanliness = (1.0 - chamber_labels.float().mean().item()) * 100.0 if len(chamber_labels) > 0 else 0
        isolate_anom_rate = isolate_labels.float().mean().item() * 100.0 if len(isolate_labels) > 0 else 0
        
        # Enrichment factor: how much denser anomalies are in isolate pool compared to global average
        enrichment_factor = (isolate_anom_rate / 100.0) / global_anomaly_rate if global_anomaly_rate > 0 else 0
        
        results.append({
            'Chamber Boundary Heuristic': name,
            'In-Chamber Nodes': num_in_chamber,
            'Isolate Pool Nodes': num_isolates,
            'In-Chamber Cleanliness (%)': f"{chamber_cleanliness:.2f}%",
            'Isolate Anomaly Rate (%)': f"{isolate_anom_rate:.2f}%",
            'Isolate Anomaly Enrichment': f"{enrichment_factor:.2f}x Global"
        })
        
    df = pd.DataFrame(results)
    print("\n==========================================================================")
    print("                    HOMOPHILY BOUNDARY RESULTS TABLE")
    print("==========================================================================")
    print(df.to_string(index=False))
    
    df.to_csv('/home/utsab/Data/EchoBasin-GAD/echo_analysis/chamber_boundary_results.csv', index=False)
    print("\nResults saved to echo_analysis/chamber_boundary_results.csv")

if __name__ == '__main__':
    evaluate_chamber_heuristics()
