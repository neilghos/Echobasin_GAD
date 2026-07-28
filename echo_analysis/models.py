"""
===============================================================================
ECHO-BASIN GRAPH ANOMALY DETECTION (GAD) - PYTORCH GEOMETRIC MODELS
===============================================================================
Uses PyTorch Geometric (torch_geometric.nn) for all GNN backbones and modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class PyGGraphSAGEEncoder(nn.Module):
    """2-Layer PyTorch Geometric GraphSAGE Encoder"""
    def __init__(self, in_dim, hidden_dim=64, out_dim=64, dropout=0.2):
        super(PyGGraphSAGEEncoder, self).__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h

class InChamberEvictor(nn.Module):
    """
    Stage 2 Part A: In-Chamber Anomaly Evictor (PyTorch).
    Evaluates representation divergence between a chamber node u and 
    its hub centroid. Nodes exceeding divergence threshold tau_evict are evicted to isolates.
    """
    def __init__(self, hidden_dim=64):
        super(InChamberEvictor, self).__init__()
        self.divergence_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def compute_evictions(self, h, chambers, tau_evict=0.50):
        evicted_nodes = set()
        for hub, chamber_nodes in chambers.items():
            if len(chamber_nodes) <= 1:
                continue
            hub_embed = h[hub].unsqueeze(0)  # [1, D]
            cand_embeds = h[chamber_nodes]   # [N_c, D]
            
            # Pair candidate node embedding with hub centroid embedding
            paired = torch.cat([cand_embeds, hub_embed.expand(len(chamber_nodes), -1)], dim=-1)
            div_scores = self.divergence_mlp(paired).squeeze(-1)
            
            # Nodes exceeding divergence threshold tau_evict are evicted
            evict_mask = div_scores > tau_evict
            for idx, is_evicted in enumerate(evict_mask.tolist()):
                if is_evicted and chamber_nodes[idx] != hub:
                    evicted_nodes.add(chamber_nodes[idx])
                    
        return evicted_nodes

class IsolateNodeRouterGNN(nn.Module):
    """
    Stage 2 Part B: Isolate Node Router (PyTorch).
    Measures affinity between isolate nodes v in S_isolate and chamber centroids c_k.
    Reassigns peripheral normal isolates back into home chambers.
    """
    def __init__(self, hidden_dim=64):
        super(IsolateNodeRouterGNN, self).__init__()
        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h, chambers, isolates, theta_reassign=0.60):
        device = h.device
        num_nodes = h.shape[0]
        prob_anomaly = torch.zeros(num_nodes, device=device, dtype=torch.float)

        if len(chambers) == 0 or len(isolates) == 0:
            isolate_list = list(isolates)
            if len(isolate_list) > 0:
                prob_anomaly[isolate_list] = 0.90
            return torch.clamp(prob_anomaly, min=1e-6, max=1.0 - 1e-6), set()

        # Compute chamber centroids c_k
        hub_list = list(chambers.keys())
        centroids = []
        for hub in hub_list:
            c_nodes = chambers[hub]
            c_embed = h[c_nodes].mean(dim=0)
            centroids.append(c_embed)
        centroid_matrix = torch.stack(centroids, dim=0)  # [K, D]
        
        # Calculate affinity for each isolate v to all centroids
        isolate_tensor = torch.tensor(list(isolates), device=device, dtype=torch.long)
        isolate_embeds = h[isolate_tensor]  # [N_iso, D]
        
        # Pairwise Cosine Similarity normalized in [0, 1]
        iso_norm = F.normalize(isolate_embeds, p=2, dim=-1)
        cent_norm = F.normalize(centroid_matrix, p=2, dim=-1)
        raw_affinity = torch.matmul(iso_norm, cent_norm.T)  # [N_iso, K] in [-1, 1]
        affinity_matrix = torch.clamp(raw_affinity, min=0.0, max=1.0) # Clamp to [0, 1]
        
        max_affinity, best_chamber_idx = affinity_matrix.max(dim=-1)
        
        # Reassignment decision: isolates with max_affinity >= theta_reassign belong to home chamber
        reassigned_mask = max_affinity >= theta_reassign
        reassigned_isolates = set(isolate_tensor[reassigned_mask].tolist())
        true_anomalies = set(isolate_tensor[~reassigned_mask].tolist())
        
        # 1. Unassigned isolates get high anomaly probability in [0.85, 1.00]
        if len(true_anomalies) > 0:
            anom_idx = isolate_tensor[~reassigned_mask]
            anom_aff = max_affinity[~reassigned_mask]
            prob_anomaly[anom_idx] = 0.85 + 0.15 * (1.0 - anom_aff)
        
        # 2. Reassigned isolates get low anomaly probability in [0.00, 0.15]
        if len(reassigned_isolates) > 0:
            reass_idx = isolate_tensor[reassigned_mask]
            reass_aff = max_affinity[reassigned_mask]
            prob_anomaly[reass_idx] = 0.15 * (1.0 - reass_aff)
            
        # 3. Chamber nodes get low base anomaly probability (0.05)
        all_chamber_nodes = set().union(*chambers.values()) - reassigned_isolates
        if len(all_chamber_nodes) > 0:
            prob_anomaly[list(all_chamber_nodes)] = 0.05
            
        # Strictly clamp output probabilities to [1e-6, 1.0 - 1e-6] for BCELoss stability
        prob_anomaly = torch.clamp(prob_anomaly, min=1e-6, max=1.0 - 1e-6)
        
        return prob_anomaly, reassigned_isolates

class EchoBasinGADModel(nn.Module):
    """
    Unified EchoBasin Graph Anomaly Detection Model (PyTorch Geometric).
    Combines Stage 1 K-Core Hub Chambers + Stage 2 Router GNN.
    """
    def __init__(self, in_dim, hidden_dim=64, out_dim=64):
        super(EchoBasinGADModel, self).__init__()
        self.encoder = PyGGraphSAGEEncoder(in_dim, hidden_dim, out_dim)
        self.evictor = InChamberEvictor(out_dim)
        self.router = IsolateNodeRouterGNN(out_dim)

    def forward(self, x, edge_index, chambers, isolates, tau_evict=0.50, theta_reassign=0.60):
        # 1. Compute PyG Node Representations H
        h = self.encoder(x, edge_index)
        
        # 2. Stage 2 Part A: Evict disguised imposters from chambers to isolates
        evicted = self.evictor.compute_evictions(h, chambers, tau_evict=tau_evict)
        updated_isolates = set(isolates).union(evicted)
        
        # 3. Stage 2 Part B: Route peripheral isolates back to home chambers
        prob_anomaly, reassigned = self.router(h, chambers, updated_isolates, theta_reassign=theta_reassign)
        
        return prob_anomaly, h
