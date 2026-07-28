import torch
import torch.nn.functional as F
import dgl
import networkx as nx
import numpy as np

class HomophilyChamberBuilder:
    """
    Stage 1 Part B: Homophily Chamber Boundary Builder around K-Core Hubs.
    
    Supports 3 Foundational & Recent Literature Heuristics:
    1. 'mhop_cosine': m-hop structural neighborhood filtered by feature Cosine Similarity threshold tau.
    2. 'ppr_gate': Personalized PageRank (PPR) random walk probability weighted by Cosine Similarity.
    3. 'adaptive_variance': Adaptive per-chamber threshold tau_k = mean_k - alpha * std_k.
    """
    def __init__(self, m_hops=2, ppr_alpha=0.15):
        self.m_hops = m_hops
        self.ppr_alpha = ppr_alpha

    def _get_ego_neighbors(self, g, hub, m_hops=2):
        """Extracts 1-hop and 2-hop neighbor nodes for a given hub."""
        succ1 = g.successors(hub)
        pred1 = g.predecessors(hub)
        hop1 = torch.unique(torch.cat([succ1, pred1, torch.tensor([hub], device=g.device)]))
        
        if m_hops == 1:
            return hop1
            
        hop2_list = [hop1]
        # Get 2-hop neighbors for top hop1 nodes
        for n in hop1[:50]: # Cap at top 50 to keep computation fast
            succ2 = g.successors(n)
            pred2 = g.predecessors(n)
            hop2_list.extend([succ2, pred2])
            
        return torch.unique(torch.cat(hop2_list))

    def build_chambers_mhop_cosine(self, g, feats, hub_indices, tau=0.60):
        feats_norm = F.normalize(feats, p=2, dim=-1)
        chambers = {}
        all_chamber_nodes = set()
        
        for hub in hub_indices.tolist():
            candidate_nodes = self._get_ego_neighbors(g, hub, m_hops=self.m_hops)
            hub_feat = feats_norm[hub]
            cand_feats = feats_norm[candidate_nodes]
            
            sims = (cand_feats * hub_feat).sum(dim=-1)
            valid_mask = sims >= tau
            
            chamber_nodes = candidate_nodes[valid_mask].tolist()
            chambers[hub] = chamber_nodes
            all_chamber_nodes.update(chamber_nodes)
            
        isolates = list(set(range(g.num_nodes())) - all_chamber_nodes)
        return chambers, isolates

    def build_chambers_ppr_gate(self, g, feats, hub_indices, ppr_tau=0.005):
        """Personalized PageRank (PPR) Random Walk Diffusion + Feature Gate."""
        feats_norm = F.normalize(feats, p=2, dim=-1)
        num_nodes = g.num_nodes()
        
        nx_g = nx.Graph(dgl.to_networkx(g.cpu()).to_undirected())
        nx_g.remove_edges_from(nx.selfloop_edges(nx_g))
        
        chambers = {}
        all_chamber_nodes = set()
        
        for hub in hub_indices.tolist():
            n_vector = {hub: 1.0}
            try:
                ppr_dict = nx.pagerank(nx_g, alpha=1.0 - self.ppr_alpha, nstart=n_vector, max_iter=50)
            except Exception:
                ppr_dict = {i: 1.0/num_nodes for i in range(num_nodes)}
                
            hub_feat = feats_norm[hub]
            cand_nodes = [node for node, ppr_score in ppr_dict.items() if ppr_score > 1e-4]
            cand_tensor = torch.tensor(cand_nodes, dtype=torch.long)
            
            sims = (feats_norm[cand_tensor] * hub_feat).sum(dim=-1)
            ppr_scores = torch.tensor([ppr_dict[n] for n in cand_nodes], dtype=torch.float)
            
            gated_scores = ppr_scores * sims
            valid_mask = gated_scores >= ppr_tau
            
            chamber_nodes = cand_tensor[valid_mask].tolist()
            chambers[hub] = chamber_nodes
            all_chamber_nodes.update(chamber_nodes)
            
        isolates = list(set(range(num_nodes)) - all_chamber_nodes)
        return chambers, isolates

    def build_chambers_adaptive_variance(self, g, feats, hub_indices, alpha_std=0.5):
        """Adaptive Per-Chamber Variance Thresholding: tau_k = mean_k - alpha * std_k"""
        feats_norm = F.normalize(feats, p=2, dim=-1)
        chambers = {}
        all_chamber_nodes = set()
        
        for hub in hub_indices.tolist():
            candidate_nodes = self._get_ego_neighbors(g, hub, m_hops=1)
            
            if len(candidate_nodes) <= 1:
                chambers[hub] = [hub]
                all_chamber_nodes.add(hub)
                continue
                
            hub_feat = feats_norm[hub]
            cand_feats = feats_norm[candidate_nodes]
            sims = (cand_feats * hub_feat).sum(dim=-1)
            
            mean_sim = sims.mean()
            std_sim = sims.std(unbiased=False)
            tau_k = max(0.2, (mean_sim - alpha_std * std_sim).item())
            
            valid_mask = sims >= tau_k
            chamber_nodes = candidate_nodes[valid_mask].tolist()
            
            chambers[hub] = chamber_nodes
            all_chamber_nodes.update(chamber_nodes)
            
        isolates = list(set(range(g.num_nodes())) - all_chamber_nodes)
        return chambers, isolates
