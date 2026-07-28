"""
===============================================================================
ECHO-BASIN GRAPH ANOMALY DETECTION (GAD) - UNIFIED PIPELINE DATALOADER
===============================================================================
Loads datasets directly from the dataset repository folders, completely copies
the UniGAD data pipeline, and outputs exact train/val/test splits for nodes and edges.
"""

import os
import sys
import random
import torch
import numpy as np
import dgl
from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split

ROOT_SEED = 3407

def set_seed(seed=ROOT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class EchoDataLoader:
    """
    Unified Dataset Loader replicating UniGAD pipeline completely.
    Supports single-graph and multi-graph GAD datasets.
    """
    def __init__(self, name='weibo', base_dir='/home/utsab/Data/EchoBasin-GAD/datasets/'):
        self.name = name
        self.base_dir = base_dir
        
        # Locate dataset path
        self.prefix, self.full_name = self._resolve_dataset_path(name, base_dir)
        self.load_graph_data()
        
    def _resolve_dataset_path(self, name, base_dir):
        possible_locations = [
            (os.path.join(base_dir, 'edge_labels/'), name + '-els'),
            (os.path.join(base_dir, 'unified/'), name),
            ('/home/utsab/Data/EchoBasin-GAD/echo_analysis/datasets/', name),
            (base_dir, name)
        ]
        for prefix, fname in possible_locations:
            if os.path.exists(os.path.join(prefix, fname)):
                return prefix, fname
        raise FileNotFoundError(f"Could not locate dataset '{name}' under base_dir '{base_dir}'")

    def load_graph_data(self):
        full_path = os.path.join(self.prefix, self.full_name)
        graphs, labels_dict = load_graphs(full_path)
        
        self.graph_list = graphs
        self.g = graphs[0]
        self.num_nodes = self.g.num_nodes()
        self.num_edges = self.g.num_edges()
        
        # Standardize feature tensor
        if 'feature' in self.g.ndata:
            self.feats = self.g.ndata['feature'].float()
        elif 'feat' in self.g.ndata:
            self.feats = self.g.ndata['feat'].float()
        else:
            raise KeyError(f"No feature tensor found in graph ndata: {list(self.g.ndata.keys())}")
            
        self.in_dim = self.feats.shape[1]
        
        # Standardize node label tensor
        self.node_labels = None
        for k in ['node_label', 'label', 'labels', 'y']:
            if k in self.g.ndata:
                self.node_labels = self.g.ndata[k].long()
                break
            if isinstance(labels_dict, dict) and k in labels_dict:
                self.node_labels = labels_dict[k].long()
                break
                
        if self.node_labels is None:
            raise ValueError(f"Could not find node label key in graph ndata: {list(self.g.ndata.keys())}")
            
        # Standardize edge label tensor (if present)
        self.edge_labels = self.g.edata['edge_label'].long() if 'edge_label' in self.g.edata else None

    def get_split_ratios(self):
        """Official UniGAD dataset train/val/test split ratios"""
        if self.name in ['tolokers', 'questions']:
            return 0.50, 0.25, 0.25
        elif self.name in ['amazon', 'yelp']:
            return 0.70, 0.10, 0.20
        elif self.name in ['uni-tsocial', 'tsocial', 'tfinance', 'reddit', 'weibo']:
            return 0.40, 0.20, 0.40
        elif 'mnist' in self.name:
            return 0.10, 0.10, 0.80
        else:
            return 0.40, 0.20, 0.40

    def get_official_splits(self, total_trials=5, trial_id=0):
        """
        Outputs exact official UniGAD train/val/test splits for nodes and edges.
        
        Returns:
            dict containing:
            - 'g': DGL Graph object
            - 'feats': Node feature matrix [N, D]
            - 'node_labels': Ground truth node anomaly labels [N]
            - 'train_mask': Tensor of training node indices
            - 'val_mask': Tensor of validation node indices
            - 'test_mask': Tensor of test node indices
            - 'train_edge_mask': Tensor of training edge indices
            - 'val_edge_mask': Tensor of validation edge indices
            - 'test_edge_mask': Tensor of test edge indices
        """
        train_ratio, val_ratio, test_ratio = self.get_split_ratios()
        seed = ROOT_SEED + total_trials * trial_id
        set_seed(seed)
        
        node_indices = list(range(self.num_nodes))
        labels = self.node_labels.cpu().numpy()
        
        # 1. Stratified Node Split (reproducing UniGAD main.py / utils.py)
        idx_train, idx_rest, y_train, y_rest = train_test_split(
            node_indices, labels, stratify=labels, train_size=train_ratio, random_state=seed, shuffle=True
        )
        idx_val, idx_test, y_val, y_test = train_test_split(
            idx_rest, y_rest, stratify=y_rest, train_size=val_ratio / (val_ratio + test_ratio), random_state=seed, shuffle=True
        )
        
        train_mask_node = torch.tensor(idx_train, dtype=self.g.idtype)
        val_mask_node = torch.tensor(idx_val, dtype=self.g.idtype)
        test_mask_node = torch.tensor(idx_test, dtype=self.g.idtype)
        
        # 2. Induced Subgraph Edge Split
        train_subg = dgl.node_subgraph(self.g, train_mask_node, store_ids=True)
        val_subg = dgl.node_subgraph(self.g, val_mask_node, store_ids=True)
        test_subg = dgl.node_subgraph(self.g, test_mask_node, store_ids=True)
        
        return {
            'g': self.g,
            'feats': self.feats,
            'node_labels': self.node_labels,
            'train_mask': train_mask_node,
            'val_mask': val_mask_node,
            'test_mask': test_mask_node,
            'train_edge_mask': train_subg.edata[dgl.EID],
            'val_edge_mask': val_subg.edata[dgl.EID],
            'test_edge_mask': test_subg.edata[dgl.EID]
        }

if __name__ == '__main__':
    print("Testing EchoDataLoader on Weibo...")
    loader = EchoDataLoader('weibo')
    splits = loader.get_official_splits(trial_id=0)
    print(f"Nodes: {loader.num_nodes} | Edges: {loader.num_edges()}")
    print(f"Test Nodes: {len(splits['test_mask'])} ({len(splits['test_mask'])/loader.num_nodes:.2%})")
    print(f"Test Node Anomalies: {splits['node_labels'][splits['test_mask']].sum().item()}")
