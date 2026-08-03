"""
===============================================================================
ECHO-BASIN GRAPH ANOMALY DETECTION (GAD) - CONFIGURABLE FEATURE SET MODEL
===============================================================================
AnomalyScoringMLP supports three feature set variants:

  'baseline'   : [h || c_closest || diff || α_max]            (3D+1 = 193)
                  Original feature set.

  'full_stats' : [h || c_closest || diff || α_max || α_mean || α_std]  (3D+3 = 195)
                  Adds global chamber landscape statistics.
                  NaN fix: α_std uses population std (correction=0) to handle K=1.

  'lean_stats' : [h || diff || α_max || α_mean || α_std]      (2D+3 = 131)
                  Drops redundant c_closest (≈ h - diff), keeps all statistics.
                  Forces MLP to use capacity on signal, not reconstruction.

α_max:  max cosine similarity to any centroid  (best chamber fit)
α_mean: mean cosine similarity across all K   (global normality)
α_std:  std of similarities across all K      (inter-basin ambiguity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


# ── Input dimension helper ────────────────────────────────────────────────────
_FEAT_DIM = {
    'baseline':   lambda D: 3 * D + 1,   # 193 for D=64
    'full_stats': lambda D: 3 * D + 3,   # 195 for D=64
    'lean_stats': lambda D: 2 * D + 3,   # 131 for D=64
}


class PyGGraphSAGEEncoder(nn.Module):
    """2-Layer PyTorch Geometric GraphSAGE Encoder"""
    def __init__(self, in_dim, hidden_dim=64, out_dim=64, dropout=0.2):
        super().__init__()
        self.conv1   = SAGEConv(in_dim, hidden_dim)
        self.conv2   = SAGEConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        return h


class AnomalyScoringMLP(nn.Module):
    """
    Learnable Differentiable Anomaly Scoring Head.

    Args:
        hidden_dim:   embedding dimension D
        feature_set:  one of 'baseline', 'full_stats', 'lean_stats'
    """
    def __init__(self, hidden_dim=64, feature_set='baseline'):
        super().__init__()
        assert feature_set in _FEAT_DIM, \
            f"feature_set must be one of {list(_FEAT_DIM.keys())}"
        self.feature_set = feature_set
        in_features = _FEAT_DIM[feature_set](hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, h, centroids, chambers):
        device    = h.device
        num_nodes = h.shape[0]

        if len(chambers) == 0:
            prob = torch.full((num_nodes,), 0.50, device=device)
            return torch.clamp(prob, 1e-6, 1 - 1e-6), \
                   torch.zeros(num_nodes, device=device)

        # ── Pairwise cosine similarity [N, K] ────────────────────────────────
        h_norm    = F.normalize(h, p=2, dim=-1)           # [N, D]
        cent_norm = F.normalize(centroids, p=2, dim=-1)   # [K, D]
        affinity  = torch.clamp(
            torch.matmul(h_norm, cent_norm.T), 0.0, 1.0)  # [N, K]

        # ── Core scalars ─────────────────────────────────────────────────────
        max_affinity, best_idx = affinity.max(dim=-1)     # [N]
        c_norm_closest         = cent_norm[best_idx]      # [N, D]
        diff_vector            = h_norm - c_norm_closest  # [N, D]

        # ── Build scoring input based on feature_set ──────────────────────────
        if self.feature_set == 'baseline':
            scoring_input = torch.cat([
                h_norm,
                c_norm_closest,
                diff_vector,
                max_affinity.unsqueeze(-1),
            ], dim=-1)                                     # [N, 3D+1]

        elif self.feature_set == 'full_stats':
            mean_affinity = affinity.mean(dim=-1)          # [N]
            # correction=0 → population std, avoids NaN when K=1
            std_affinity  = affinity.std(dim=-1, correction=0)  # [N]
            scoring_input = torch.cat([
                h_norm,
                c_norm_closest,
                diff_vector,
                max_affinity.unsqueeze(-1),
                mean_affinity.unsqueeze(-1),
                std_affinity.unsqueeze(-1),
            ], dim=-1)                                     # [N, 3D+3]

        else:  # lean_stats — drop c_norm_closest
            mean_affinity = affinity.mean(dim=-1)          # [N]
            std_affinity  = affinity.std(dim=-1, correction=0)  # [N]
            scoring_input = torch.cat([
                h_norm,
                diff_vector,
                max_affinity.unsqueeze(-1),
                mean_affinity.unsqueeze(-1),
                std_affinity.unsqueeze(-1),
            ], dim=-1)                                     # [N, 2D+3]

        # ── Score ─────────────────────────────────────────────────────────────
        raw    = self.head(scoring_input).squeeze(-1)      # [N]
        prob   = torch.clamp(raw, 1e-6, 1 - 1e-6)
        return prob, max_affinity


class EchoBasinGADModel(nn.Module):
    """
    EchoBasin GAD Model.

    Args:
        feature_set: 'baseline' | 'full_stats' | 'lean_stats'
                     Controls the AnomalyScoringMLP input feature set.
    """
    def __init__(self, in_dim, hidden_dim=64, out_dim=64,
                 feature_set='baseline', is_large_graph=False):
        super().__init__()
        self.encoder      = PyGGraphSAGEEncoder(in_dim, hidden_dim, out_dim)
        self.scorer_head  = AnomalyScoringMLP(out_dim, feature_set=feature_set)

    def forward(self, x, edge_index, chambers, isolates, **kwargs):
        device = x.device
        h      = self.encoder(x, edge_index)

        if len(chambers) == 0:
            prob = torch.full((x.shape[0],), 0.50, device=device)
            return torch.clamp(prob, 1e-6, 1 - 1e-6), h

        # ── Degree-weighted chamber centroids ─────────────────────────────────
        num_nodes = h.shape[0]
        deg = torch.zeros(num_nodes, device=device)
        deg.index_add_(0, edge_index[1],
                       torch.ones(edge_index.shape[1], device=device))

        centroids = []
        for hub in chambers:
            c_t  = torch.tensor(chambers[hub], device=device, dtype=torch.long)
            c_e  = h[c_t]
            c_d  = deg[c_t].unsqueeze(-1)
            w    = c_d / torch.clamp(c_d.sum(), min=1e-6)
            centroids.append((c_e * w).sum(0))
        centroid_matrix = torch.stack(centroids, dim=0)   # [K, D]

        prob, max_affinity = self.scorer_head(h, centroid_matrix, chambers)
        return prob, h
