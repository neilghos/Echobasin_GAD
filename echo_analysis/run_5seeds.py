"""
===============================================================================
ECHO-BASIN GAD - 5-SEED DETERMINISTIC BENCHMARK RUNNER (run_5seeds.py)
===============================================================================
Runs 5-seed deterministic evaluation across [42, 100, 2024, 777, 999] to compute
Mean +- Std for AUROC, AUPRC, and Macro-F1 across all 6 benchmark datasets.
Eliminates seed noise for ICLR/NeurIPS paper submission.
"""

import sys
import os
import argparse
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))

from dataloader import EchoDataLoader
from hub import get_kcore_hubs, build_homophily_boundaries, derive_k_ideal
from train import train_echobasin
from eval import evaluate

ALL_DATASETS = ['weibo', 'reddit', 'amazon', 'yelp', 'tolokers', 'questions']
SEEDS = [42, 100, 2024, 777, 999]

def set_seed(seed):
    """Sets deterministic random seeds across Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_dataset_5seeds(dataset_name, epochs=100):
    print(f"\n==========================================================================")
    print(f"   5-SEED DETERMINISTIC RUNNER ON: {dataset_name.upper()}")
    print(f"   Seeds: {SEEDS} | Epochs: {epochs}")
    print(f"==========================================================================")
    
    auroc_list, auprc_list, f1_list = [], [], []
    
    loader = EchoDataLoader(dataset_name)
    
    for trial_idx, seed in enumerate(SEEDS):
        set_seed(seed)
        
        # Load official UniGAD split for trial
        splits = loader.get_official_splits(trial_id=(trial_idx % 5))
        g = splits['g']
        feats = splits['feats']
        labels = splits['node_labels']
        train_mask = splits['train_mask']
        val_mask = splits['val_mask']
        test_mask = splits['test_mask']
        
        num_nodes = g.num_nodes()
        num_edges = g.num_edges()
        
        # Stage 1 Hub & Chamber Discovery
        k_ideal = derive_k_ideal(g)
        hubs, scores, norm_coreness = get_kcore_hubs(g, top_k=k_ideal)
        chambers, isolates, stats = build_homophily_boundaries(g, feats, hubs)
        
        # Train Model
        trained_model, best_val_thresh, val_metrics, x_train = train_echobasin(
            g=g,
            feats=feats,
            node_labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            chambers=chambers,
            isolates=isolates,
            in_dim=loader.in_dim,
            epochs=epochs,
            lr=0.01
        )
        
        # Test Evaluation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0).to(device)
        
        trained_model.eval()
        with torch.no_grad():
            test_probs, _ = trained_model(x_train, edge_index, chambers, isolates)
            
        test_probs_np = test_probs[test_mask].cpu().numpy()
        test_labels_np = labels[test_mask].cpu().numpy()
        
        test_metrics = evaluate(test_labels_np, test_probs_np)
        
        auroc_list.append(test_metrics['AUROC'])
        auprc_list.append(test_metrics['AUPRC'])
        f1_list.append(test_metrics['Macro-F1'])
        
        print(f"  [Seed {seed:<4} | Trial {trial_idx}] Test AUROC: {test_metrics['AUROC']:.4f} | AUPRC: {test_metrics['AUPRC']:.4f} | Macro-F1: {test_metrics['Macro-F1']:.4f}")
        
    auroc_mean, auroc_std = np.mean(auroc_list), np.std(auroc_list)
    auprc_mean, auprc_std = np.mean(auprc_list), np.std(auprc_list)
    f1_mean, f1_std = np.mean(f1_list), np.std(f1_list)
    
    print(f"\n--- 5-SEED FINAL SUMMARY ({dataset_name.upper()}) ---")
    print(f"  • Test AUROC:    {auroc_mean:.4f} +- {auroc_std:.4f}")
    print(f"  • Test AUPRC:    {auprc_mean:.4f} +- {auprc_std:.4f}")
    print(f"  • Test Macro-F1: {f1_mean:.4f} +- {f1_std:.4f}")
    print("==========================================================================")
    
    return {
        'Dataset': dataset_name,
        'AUROC_Mean': auroc_mean,
        'AUROC_Std': auroc_std,
        'AUPRC_Mean': auprc_mean,
        'AUPRC_Std': auprc_std,
        'F1_Mean': f1_mean,
        'F1_Std': f1_std,
        'AUROC_Str': f"{auroc_mean:.4f} +- {auroc_std:.4f}",
        'AUPRC_Str': f"{auprc_mean:.4f} +- {auprc_std:.4f}",
        'F1_Str': f"{f1_mean:.4f} +- {f1_std:.4f}"
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='weibo', help='Dataset name (or all)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    args = parser.parse_args()
    
    if args.dataset != 'all':
        run_dataset_5seeds(args.dataset, epochs=args.epochs)
    else:
        results = []
        dset_pbar = tqdm(ALL_DATASETS, desc="Evaluating Datasets across 5 Seeds", leave=True)
        for dname in dset_pbar:
            dset_pbar.set_postfix({'Current': dname})
            res = run_dataset_5seeds(dname, epochs=args.epochs)
            if res is not None:
                results.append(res)
                
        if len(results) > 0:
            df = pd.DataFrame(results)
            print("\n==========================================================================")
            print("        ECHO-BASIN GAD 5-SEED DETERMINISTIC RESULTS MATRIX")
            print("==========================================================================")
            print(df[['Dataset', 'AUROC_Str', 'AUPRC_Str', 'F1_Str']].to_string(index=False))
            df.to_csv('/home/utsab/Data/EchoBasin-GAD/echo_analysis/echobasin_5seed_results.csv', index=False)
            print("\n5-Seed Results saved to echo_analysis/echobasin_5seed_results.csv")

if __name__ == '__main__':
    main()
