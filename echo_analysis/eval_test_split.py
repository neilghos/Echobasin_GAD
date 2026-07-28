import sys
import os
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.dirname(__file__))

from utils import Dataset
from hub import get_kcore_hubs, build_homophily_boundaries, derive_k_ideal
from e2e_models import get_best_f1, roc_auc_score, average_precision_score

def eval_stage1_on_official_test_split(dataset_name='weibo', trial_id=0):
    print(f"\n==========================================================================")
    print(f"   EVALUATING STAGE 1 HEURISTIC ON OFFICIAL UNIGAD TEST SET SPLIT")
    print(f"   Dataset: {dataset_name} | Trial ID: {trial_id}")
    print(f"==========================================================================")
    
    if dataset_name in ['reddit', 'weibo', 'amazon', 'yelp', 'tfinance', 'tolokers', 'questions']:
        dataset = Dataset(dataset_name + '-els', prefix='/home/utsab/Data/EchoBasin-GAD/datasets/edge_labels/', sp_type='star+norm', labels_have='ne')
    else:
        dataset = Dataset(dataset_name)
        
    dataset.prepare_dataset(total_trials=5)
    dataset.make_sp_matrix_graph_list(khop=1, load_kg=True)
    
    g = dataset.graph_list[0]
    feats = g.ndata['feature']
    labels = dataset.node_label[0]
    
    # Get official UniGAD train/val/test masks for this trial
    train_loader, val_loader, test_loader = dataset.get_graph_and_sp_dataloaders(batch_size=1, trial_id=trial_id)
    test_mask = dataset.test_mask_node_cur # Official boolean test mask!
    
    test_node_indices = test_mask.nonzero().squeeze().cpu()
    test_labels = labels[test_node_indices].cpu().numpy()
    
    print(f"Graph Total Nodes: {g.num_nodes()}")
    print(f"Official Test Set Node Count: {len(test_node_indices)} ({len(test_node_indices)/g.num_nodes():.2%} of graph)")
    print(f"Official Test Set Anomalies: {test_labels.sum()} ({test_labels.mean():.2%})")
    
    # 1. Run Stage 1 K-Core Hub & Homophily Chamber Boundary
    k_ideal = derive_k_ideal(g)
    hubs, scores, norm_coreness = get_kcore_hubs(g, top_k=k_ideal)
    chambers, isolates, stats = build_homophily_boundaries(g, feats, hubs, tau=0.60, m_hops=2)
    
    # 2. Continuous Anomaly Probability Scores based on in_homophily check
    num_nodes = g.num_nodes()
    in_homophily = torch.zeros(num_nodes, dtype=torch.bool)
    
    in_chamber_nodes = set()
    for c_nodes in chambers.values():
        in_chamber_nodes.update(c_nodes)
    
    in_homophily[list(in_chamber_nodes)] = True
    
    # Anomaly Probability P(anomaly): 0.10 if in homophily chamber, 0.90 if isolate
    probs = torch.where(in_homophily, 0.10, 0.90)
    
    # 3. Evaluate ONLY on Official Test Set Nodes!
    test_probs = probs[test_node_indices].cpu().numpy()
    
    macro_f1, best_thresh = get_best_f1(test_labels, test_probs)
    auroc = roc_auc_score(test_labels, test_probs)
    auprc = average_precision_score(test_labels, test_probs)
    
    print("\n==========================================================================")
    print("        OFFICIAL UNIGAD TEST SET EVALUATION METRICS")
    print("==========================================================================")
    print(f"  • Official Test Set AUROC:      {auroc:.4f}")
    print(f"  • Official Test Set AUPRC (AP): {auprc:.4f}")
    print(f"  • Official Test Set Macro-F1:   {macro_f1:.4f}  (Optimal threshold: {best_thresh:.2f})")
    print("==========================================================================")
    
    return {
        'Test AUROC': auroc,
        'Test AUPRC': auprc,
        'Test MacroF1': macro_f1
    }

if __name__ == '__main__':
    eval_stage1_on_official_test_split('reddit', trial_id=0)
