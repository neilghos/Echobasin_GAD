"""
===============================================================================
ECHO-BASIN GRAPH ANOMALY DETECTION (GAD) - STABILIZED TRAINER (train.py)
===============================================================================
Trains EchoBasinGADModel (AnomalyScoringMLP Head) using:
1. Stabilized Weighted Focal Loss (clamped pos_weight <= 5.0, gamma=2.0)
2. Soft Chamber Homophily Alignment Loss (scale = 0.01)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models import EchoBasinGADModel
from eval import get_best_f1, evaluate

def compute_focal_loss(pred_probs, labels, pos_weight=1.0, gamma=2.0):
    """
    Stabilized Weighted Focal Cross-Entropy Loss (gamma=2.0).
    Clamps pos_weight <= 5.0 to prevent gradient explosion on highly imbalanced graphs.
    """
    labels_float = labels.float()
    
    # Focal terms: (1 - p)^gamma for positive, p^gamma for negative
    p_t = pred_probs * labels_float + (1.0 - pred_probs) * (1.0 - labels_float)
    focal_weight = (1.0 - p_t) ** gamma
    
    # Class weighting with upper cap to prevent gradient explosion
    clamped_pos_weight = min(float(pos_weight), 5.0)
    weight_t = clamped_pos_weight * labels_float + 1.0 * (1.0 - labels_float)
    
    bce = - (labels_float * torch.log(pred_probs) + (1.0 - labels_float) * torch.log(1.0 - pred_probs))
    loss = (focal_weight * weight_t * bce).mean()
    return loss

def compute_homophily_alignment_loss(h, chambers, pred_probs):
    """
    Unsupervised Chamber Homophily Alignment Loss over all nodes:
    L_Homophily = Mean( (P(v) - (1.0 - alpha_v))^2 )
    """
    if len(chambers) == 0:
        return torch.tensor(0.0, device=h.device)
        
    hub_list = list(chambers.keys())
    centroids = []
    for hub in hub_list:
        c_nodes = chambers[hub]
        c_embed = h[c_nodes].mean(dim=0)
        centroids.append(c_embed)
    centroid_matrix = torch.stack(centroids, dim=0)  # [K, D]
    
    h_norm = F.normalize(h, p=2, dim=-1)
    cent_norm = F.normalize(centroid_matrix, p=2, dim=-1)
    raw_affinity = torch.matmul(h_norm, cent_norm.T)
    max_affinity, _ = torch.clamp(raw_affinity, min=0.0, max=1.0).max(dim=-1)
    
    target_score = 1.0 - max_affinity
    loss = F.mse_loss(pred_probs, target_score)
    return loss

def train_echobasin(g, feats, node_labels, train_mask, val_mask, chambers, isolates, in_dim, epochs=100, lr=0.01, weight_decay=1e-4):
    """
    Trains EchoBasin model and returns (trained_model, best_val_threshold, val_metrics).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare PyG edge_index with OOM protection for >10M edge graphs
    src, dst = g.edges()
    num_edges = len(src)
    
    if num_edges > 10000000:
        perm = torch.randperm(num_edges)[:3000000]
        src_samp, dst_samp = src[perm], dst[perm]
        edge_index = torch.stack([src_samp, dst_samp], dim=0).to(device)
    else:
        edge_index = torch.stack([src, dst], dim=0).to(device)
        
    x = feats.to(device)
    labels = node_labels.to(device)
    
    # Class imbalance weight
    train_labels = labels[train_mask]
    num_pos = max(train_labels.sum().item(), 1)
    num_neg = max(len(train_labels) - num_pos, 1)
    pos_weight = float(num_neg / num_pos)
    
    # Instantiate Model & Optimizer
    is_large = (num_edges >= 10000000)
    model = EchoBasinGADModel(in_dim=in_dim, hidden_dim=64, out_dim=64, is_large_graph=is_large).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_f1 = -1.0
    best_val_threshold = 0.50
    best_val_metrics = None
    
    pbar = tqdm(range(1, epochs + 1), desc="Training Epochs", leave=True)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass
        pred_probs, h = model(x, edge_index, chambers, isolates)
        
        # 1. Stabilized Weighted Focal Loss on Training Mask
        loss_focal = compute_focal_loss(pred_probs[train_mask], train_labels, pos_weight=pos_weight, gamma=2.0)
        
        # 2. Soft Chamber Homophily Alignment Loss (scaled to 0.01)
        loss_homo = compute_homophily_alignment_loss(h, chambers, pred_probs)
        
        # Total Loss
        total_loss = loss_focal + 0.01 * loss_homo
        total_loss.backward()
        optimizer.step()
        
        # Validation Evaluation
        model.eval()
        with torch.no_grad():
            val_probs, _ = model(x, edge_index, chambers, isolates)
            val_probs_np = val_probs[val_mask].cpu().numpy()
            val_labels_np = labels[val_mask].cpu().numpy()
            
            val_f1, val_thresh = get_best_f1(val_labels_np, val_probs_np)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_threshold = val_thresh
                best_val_metrics = evaluate(val_labels_np, val_probs_np)
                
        pbar.set_postfix({'Loss': f"{total_loss.item():.4f}", 'Val F1': f"{val_f1:.4f}"})
                
    return model, best_val_threshold, best_val_metrics
