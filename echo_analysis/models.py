"""
===============================================================================
ECHO-BASIN GRAPH ANOMALY DETECTION (GAD) - DEGREE-WEIGHTED CENTROIDS MODEL
===============================================================================
Features Degree-Weighted Chamber Centroid Readout:
c_k = Sum_{u in H_k} ( deg(u) / Sum_{w in H_k} deg(w) ) * h_u

Gives high-degree core hubs higher weight in defining chamber centroids, 
making centroids highly resilient to peripheral boundary noise!
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

class AnomalyScoringMLP(nn.Module):
    """
    Learnable Differentiable Anomaly Scoring Head.
    Input: [h_norm || c_norm_closest || (h_norm - c_norm_closest) || alpha_v] (dim = 3*D + 1)
    Output: Calibrated Anomaly Probability P(v) in [0, 1]
    """
    def __init__(self, hidden_dim=64):
        super(AnomalyScoringMLP, self).__init__()
        in_features = hidden_dim * 3 + 1  # 64*3 + 1 = 193
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, h, centroids, chambers):
        device = h.device
        num_nodes, dim = h.shape
        
        if len(chambers) == 0:
            prob_anomaly = torch.full((num_nodes,), 0.50, device=device)
            return torch.clamp(prob_anomaly, min=1e-6, max=1.0 - 1e-6)

        # 1. Compute Pairwise Cosine Similarity Matrix between all N nodes and K centroids
        h_norm = F.normalize(h, p=2, dim=-1)                   # [N, D]
        cent_norm = F.normalize(centroids, p=2, dim=-1)        # [K, D]
        raw_affinity = torch.matmul(h_norm, cent_norm.T)       # [N, K]
        affinity_matrix = torch.clamp(raw_affinity, min=0.0, max=1.0)
        
        # 2. Extract closest centroid vector and max affinity alpha_v for every node v
        max_affinity, best_centroid_idx = affinity_matrix.max(dim=-1)  # [N]
        c_norm_closest = cent_norm[best_centroid_idx]                  # [N, D]
        
        # 3. Construct Feature Vector for Scoring Head
        diff_vector = h_norm - c_norm_closest                          # [N, D]
        alpha_tensor = max_affinity.unsqueeze(-1)                      # [N, 1]
        
        scoring_input = torch.cat([h_norm, c_norm_closest, diff_vector, alpha_tensor], dim=-1) # [N, 3*D + 1]
        
        # 4. Predict Anomaly Probability P(v)
        raw_scores = self.head(scoring_input).squeeze(-1)              # [N]
        prob_anomaly = torch.clamp(raw_scores, min=1e-6, max=1.0 - 1e-6)
        
        return prob_anomaly, max_affinity

class EchoBasinGADModel(nn.Module):
    """
    Unified EchoBasin Graph Anomaly Detection Model with Degree-Weighted Chamber Centroids.
    """
    def __init__(self, in_dim, hidden_dim=64, out_dim=64, is_large_graph=False):
        super(EchoBasinGADModel, self).__init__()
        self.encoder = PyGGraphSAGEEncoder(in_dim, hidden_dim, out_dim)
        self.scorer_head = AnomalyScoringMLP(out_dim)

    def forward(self, x, edge_index, chambers, isolates, **kwargs):
        device = x.device
        
        # 1. Compute Node Representations H via GNN Encoder
        h = self.encoder(x, edge_index)
        
        if len(chambers) == 0:
            prob_anomaly = torch.full((x.shape[0],), 0.50, device=device)
            return torch.clamp(prob_anomaly, min=1e-6, max=1.0 - 1e-6), h
            
        # 2. Compute Node Degrees for Degree-Weighted Chamber Readout
        num_nodes = h.shape[0]
        deg = torch.zeros(num_nodes, device=device)
        deg.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], device=device))
        
        # 3. Compute Degree-Weighted Chamber Centroids c_k = Sum_{u in H_k} (deg(u)/Sum deg) * h_u
        hub_list = list(chambers.keys())
        centroids = []
        for hub in hub_list:
            c_nodes = chambers[hub]
            c_nodes_tensor = torch.tensor(c_nodes, device=device, dtype=torch.long)
            c_embeds = h[c_nodes_tensor]                                  # [M, D]
            c_degs = deg[c_nodes_tensor].unsqueeze(-1)                   # [M, 1]
            deg_sum = torch.clamp(c_degs.sum(), min=1e-6)
            c_weights = c_degs / deg_sum                                  # [M, 1]
            weighted_centroid = (c_embeds * c_weights).sum(dim=0)         # [D]
            centroids.append(weighted_centroid)
            
        centroid_matrix = torch.stack(centroids, dim=0)                 # [K, D]
        
        # 4. Predict End-to-End Calibrated Anomaly Probabilities P(v) via SOTA Head
        prob_anomaly, max_affinity = self.scorer_head(h, centroid_matrix, chambers)
        
        return prob_anomaly, h
