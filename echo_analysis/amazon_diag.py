"""
===============================================================================
AMAZON COLLAPSE DIAGNOSTIC SCRIPT
===============================================================================
Runs epoch-by-epoch probe of every neural submodule to find exactly which one collapses.
Checks:
  1. GNN Encoder output statistics (mean, std, norm)
  2. Chamber Centroid statistics
  3. Cosine Affinity Distribution (alpha_v for anomalies vs normals)
  4. AnomalyScoringMLP Head input distribution + predicted probability distribution
  5. Gradient norms per module
  6. Per-class predicted probability at epoch 1, 10, 50, 100
"""

import sys, os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append('/home/utsab/Data/EchoBasin-GAD/echo_analysis')

from dataloader import EchoDataLoader
from hub import get_kcore_hubs, build_homophily_boundaries, derive_k_ideal
from models import EchoBasinGADModel

def run_diagnostic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("  AMAZON COLLAPSE DIAGNOSTIC")
    print(f"{'='*70}\n")

    # 1. Load Amazon
    loader = EchoDataLoader('amazon')
    splits = loader.get_official_splits(trial_id=0)
    g = splits['g']
    feats = splits['feats']
    labels = splits['node_labels']
    train_mask = splits['train_mask']

    # 2. Build Chambers (adaptive - no hardcoded tau/m_hops)
    k_ideal = derive_k_ideal(g)
    hubs, scores, norm_coreness = get_kcore_hubs(g, top_k=k_ideal)
    chambers, isolates, stats = build_homophily_boundaries(g, feats, hubs)
    
    print(f"  K_ideal Hubs: {k_ideal}")
    print(f"  In-Chamber: {stats['num_in_chamber_nodes']} nodes | Isolates: {stats['num_isolate_nodes']} nodes")
    print(f"  m_hops={stats['m_hops_used']}, tau={stats['tau_used']}")

    # 3. Build edge_index (sampled for Amazon)
    src, dst = g.edges()
    num_edges = len(src)
    if num_edges > 10000000:
        perm = torch.randperm(num_edges)[:3000000]
        edge_index = torch.stack([src[perm], dst[perm]], dim=0).to(device)
    else:
        edge_index = torch.stack([src, dst], dim=0).to(device)
    
    x = feats.to(device)
    labels_dev = labels.to(device)
    train_labels = labels_dev[train_mask]
    
    num_pos = max(train_labels.sum().item(), 1)
    num_neg = max(len(train_labels) - num_pos, 1)
    pos_weight = float(num_neg / num_pos)
    print(f"\n  Train: {len(train_labels)} nodes | Positives: {int(num_pos)} | pos_weight: {pos_weight:.2f}")

    # 4. Identify anomaly vs normal nodes
    anom_nodes = (labels_dev == 1).nonzero(as_tuple=True)[0]
    norm_nodes  = (labels_dev == 0).nonzero(as_tuple=True)[0]
    print(f"  Total Anomaly Nodes: {len(anom_nodes)} | Normal Nodes: {len(norm_nodes)}")

    # 5. Instantiate model
    model = EchoBasinGADModel(in_dim=loader.in_dim, hidden_dim=64, out_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    check_epochs = [1, 5, 10, 30, 50, 100]

    print(f"\n{'='*70}")
    print(f"  {'Epoch':>6} | {'Loss':>8} | {'GNN_h_std':>10} | {'Aff_anom':>10} | {'Aff_norm':>10} | {'P_anom':>8} | {'P_norm':>8} | {'Grad_enc':>10} | {'Grad_mlp':>10}")
    print(f"  {'-'*100}")

    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()

        pred_probs, h = model(x, edge_index, chambers, isolates)

        train_labels_float = train_labels.float()
        weights = torch.ones_like(train_labels_float)
        weights[train_labels == 1] = pos_weight
        loss = F.binary_cross_entropy(pred_probs[train_mask], train_labels_float, weight=weights)
        loss.backward()
        
        # Gradient norms
        enc_grad = sum(p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None)
        mlp_grad = sum(p.grad.norm().item() for p in model.scorer_head.parameters() if p.grad is not None) if hasattr(model, 'scorer_head') else 0.0

        optimizer.step()

        if epoch in check_epochs:
            model.eval()
            with torch.no_grad():
                pred_probs_eval, h_eval = model(x, edge_index, chambers, isolates)

                # GNN encoder embedding stats
                gnn_std = h_eval.std().item()

                # Chamber centroids
                hub_list = list(chambers.keys())[:100]
                centroids = torch.stack([h_eval[chambers[hub]].mean(dim=0) for hub in hub_list], dim=0)

                # Affinity distribution
                h_norm = F.normalize(h_eval, p=2, dim=-1)
                c_norm = F.normalize(centroids, p=2, dim=-1)
                aff = torch.matmul(h_norm, c_norm.T).clamp(0, 1).max(dim=-1).values

                aff_anom = aff[anom_nodes].mean().item()
                aff_norm = aff[norm_nodes].mean().item()

                # Predicted probability per class
                p_anom = pred_probs_eval[anom_nodes].mean().item()
                p_norm = pred_probs_eval[norm_nodes].mean().item()

            print(f"  {epoch:>6} | {loss.item():>8.4f} | {gnn_std:>10.4f} | {aff_anom:>10.4f} | {aff_norm:>10.4f} | {p_anom:>8.4f} | {p_norm:>8.4f} | {enc_grad:>10.4f} | {mlp_grad:>10.4f}")

    print(f"\n{'='*70}")
    print("  KEY DIAGNOSTICS:")
    print("  - If Aff_anom ≈ Aff_norm: Chamber oversaturation killing signal")
    print("  - If P_anom ≈ P_norm: MLP head collapsed (scores normal=anomaly)")
    print("  - If Grad_enc ≈ 0: Dead GNN encoder (no gradient flowing)")
    print("  - If Grad_mlp ≈ 0: Dead MLP head (Sigmoid saturation)")
    print("  - If GNN_h_std ≈ 0: GNN oversmoothing (all nodes same embedding)")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    run_diagnostic()
