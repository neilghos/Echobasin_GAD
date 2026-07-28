"""
===============================================================================
ECHO-BASIN GRAPH ANOMALY DETECTION (GAD) - TRAINER (train.py)
===============================================================================
Trains EchoBasinGADModel on the training set mask using the dual-objective loss
(Weighted BCE + Chamber Compactness Loss). Evaluates on validation set mask 
to tune the optimal Macro-F1 threshold (thresh_val).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models import EchoBasinGADModel
from eval import get_best_f1, evaluate

def compute_chamber_loss(h, chambers):
    """Computes unsupervised chamber compactness loss over normal hub communities."""
    if len(chambers) == 0:
        return torch.tensor(0.0, device=h.device)
        
    loss = 0.0
    count = 0
    for hub, c_nodes in chambers.items():
        if len(c_nodes) <= 1:
            continue
        c_embed = h[c_nodes].mean(dim=0, keepdim=True) # Chamber readout centroid
        c_norm = F.normalize(c_embed, p=2, dim=-1)
        node_embeds = F.normalize(h[c_nodes], p=2, dim=-1)
        
        # 1 - CosineSim(h_u, c_k)
        sims = (node_embeds * c_norm).sum(dim=-1)
        loss += (1.0 - sims).mean()
        count += 1
        
    return loss / max(count, 1)

def train_echobasin(g, feats, node_labels, train_mask, val_mask, chambers, isolates, in_dim, epochs=30, lr=0.01, weight_decay=1e-4):
    """
    Trains EchoBasin model and returns (trained_model, best_val_threshold, val_metrics).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare PyG edge_index
    src, dst = g.edges()
    edge_index = torch.stack([src, dst], dim=0).to(device)
    x = feats.to(device)
    labels = node_labels.to(device)
    
    # Instantiate Model & Optimizer
    model = EchoBasinGADModel(in_dim=in_dim, hidden_dim=64, out_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Class imbalance weighting for BCE
    train_labels = labels[train_mask]
    num_pos = max(train_labels.sum().item(), 1)
    num_neg = max(len(train_labels) - num_pos, 1)
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    bce_loss_fn = nn.BCELoss(reduction='mean')
    
    best_val_f1 = -1.0
    best_val_threshold = 0.50
    best_val_metrics = None
    
    train_labels_float = train_labels.float()
    
    pbar = tqdm(range(1, epochs + 1), desc="Training Epochs", leave=True)
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # Forward Pass
        pred_probs, h = model(x, edge_index, chambers, isolates, tau_evict=0.50, theta_reassign=0.60)
        
        # 1. Weighted BCE Loss on Training Mask
        loss_bce = bce_loss_fn(pred_probs[train_mask], train_labels_float)
        
        # 2. Chamber Compactness Loss
        loss_chamber = compute_chamber_loss(h, chambers)
        
        # Total Dual-Objective Loss
        total_loss = loss_bce + 0.10 * loss_chamber
        total_loss.backward()
        optimizer.step()
        
        # Validation Evaluation
        model.eval()
        with torch.no_grad():
            val_probs, _ = model(x, edge_index, chambers, isolates, tau_evict=0.50, theta_reassign=0.60)
            val_probs_np = val_probs[val_mask].cpu().numpy()
            val_labels_np = labels[val_mask].cpu().numpy()
            
            val_f1, val_thresh = get_best_f1(val_labels_np, val_probs_np)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_threshold = val_thresh
                best_val_metrics = evaluate(val_labels_np, val_probs_np)
                
        pbar.set_postfix({'Loss': f"{total_loss.item():.4f}", 'Val F1': f"{val_f1:.4f}"})
                
    return model, best_val_threshold, best_val_metrics
