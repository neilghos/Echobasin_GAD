"""
===============================================================================
ECHO-BASIN GRAPH ANOMALY DETECTION (GAD) - MAIN PIPELINE RUNNER (main.py)
===============================================================================
Master entry point:
1. Loads dataset and official splits via dataloader.py
2. Discovers Stage 1 Hubs & Homophily Chambers via hub.py
3. Trains EchoBasin model & tunes validation threshold via train.py
4. Evaluates final performance on official Test Set Split via eva.py
"""

import sys
import os
import argparse
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

def run_pipeline_on_dataset(dataset_name, trial_id=0, epochs=30):
    print(f"\n==========================================================================")
    print(f"   RUNNING ECHO-BASIN GAD PIPELINE ON: {dataset_name.upper()}")
    print(f"   Trial ID: {trial_id} | Official UniGAD Test Set Split")
    print(f"==========================================================================")
    
    try:
        # 1. Load Data & Official UniGAD Splits
        loader = EchoDataLoader(dataset_name)
        splits = loader.get_official_splits(trial_id=trial_id)
        
        g = splits['g']
        feats = splits['feats']
        labels = splits['node_labels']
        train_mask = splits['train_mask']
        val_mask = splits['val_mask']
        test_mask = splits['test_mask']
        
        num_nodes = g.num_nodes()
        num_edges = g.num_edges()
        
        # 2. Stage 1 Hub & Homophily Chamber Discovery via hub.py
        k_ideal = derive_k_ideal(g)
        hubs, scores, norm_coreness = get_kcore_hubs(g, top_k=k_ideal)
        chambers, isolates, stats = build_homophily_boundaries(g, feats, hubs)
        
        print(f"Graph: {num_nodes} Nodes, {num_edges} Edges | K_ideal Hubs: {k_ideal}")
        print(f"In-Chamber Pool: {stats['num_in_chamber_nodes']} nodes | Isolate Pool: {stats['num_isolate_nodes']} nodes")
        
        # 3. Train EchoBasin Model & Tune Validation Threshold via train.py
        print("\nTraining EchoBasin GNN Model...")
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
        
        print(f"Optimal Validation Threshold Tuned: {best_val_thresh:.2f} (Val Macro-F1: {val_metrics['Macro-F1']:.4f})")
        
        # 4. Final Test Set Evaluation via eva.py
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src, dst = g.edges()
        edge_index = torch.stack([src, dst], dim=0).to(device)
        
        trained_model.eval()
        with torch.no_grad():
            test_probs, _ = trained_model(
                x_train,
                edge_index,
                chambers,
                isolates
            )
            
        test_probs_np = test_probs[test_mask].cpu().numpy()
        test_labels_np = labels[test_mask].cpu().numpy()
        
        test_metrics = evaluate(test_labels_np, test_probs_np)
        
        print("\n--- OFFICIAL TEST SET EVALUATION METRICS ---")
        print(f"  • Test Set AUROC:      {test_metrics['AUROC']:.4f}")
        print(f"  • Test Set AUPRC (AP): {test_metrics['AUPRC']:.4f}")
        print(f"  • Test Set Macro-F1:   {test_metrics['Macro-F1']:.4f} (Validation-Tuned Thresh: {best_val_thresh:.2f})")
        print("==========================================================================")
        
        return {
            'Dataset': dataset_name,
            'Status': 'SUCCESS',
            'Nodes': num_nodes,
            'Edges': num_edges,
            'Test Nodes': len(test_mask),
            'Derived K': k_ideal,
            'Val Thresh': f"{best_val_thresh:.2f}",
            'Test AUROC': f"{test_metrics['AUROC']:.4f}",
            'Test AUPRC': f"{test_metrics['AUPRC']:.4f}",
            'Test Macro-F1': f"{test_metrics['Macro-F1']:.4f}"
        }
        
    except Exception as e:
        print(f"❌ Failed processing '{dataset_name}': {e}")
        return {
            'Dataset': dataset_name,
            'Status': f'FAILED: {str(e)[:30]}',
            'Nodes': 'N/A',
            'Edges': 'N/A',
            'Test Nodes': 'N/A',
            'Derived K': 'N/A',
            'Val Thresh': 'N/A',
            'Test AUROC': 'N/A',
            'Test AUPRC': 'N/A',
            'Test Macro-F1': 'N/A'
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='weibo', help='Dataset name (or all)')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials')
    args = parser.parse_args()
    
    if args.dataset != 'all':
        run_pipeline_on_dataset(args.dataset, trial_id=0, epochs=args.epochs)
    else:
        results = []
        dset_pbar = tqdm(ALL_DATASETS, desc="Evaluating Datasets", leave=True)
        for dname in dset_pbar:
            dset_pbar.set_postfix({'Current': dname})
            res = run_pipeline_on_dataset(dname, trial_id=0, epochs=args.epochs)
            if res is not None:
                results.append(res)
        if len(results) > 0:
            df = pd.DataFrame(results)
            print("\n==========================================================================")
            print("        ECHO-BASIN GAD FULL BENCHMARK RESULTS MATRIX")
            print("==========================================================================")
            print(df.to_string(index=False))
            df.to_csv('/home/utsab/Data/EchoBasin-GAD/echo_analysis/echobasin_full_results.csv', index=False)
            print("\nResults saved to echo_analysis/echobasin_full_results.csv")

if __name__ == '__main__':
    main()
