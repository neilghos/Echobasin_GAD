"""
===============================================================================
ECHO-BASIN STANDALONE EVALUATOR (eval.py)
===============================================================================
Compact, standalone evaluator copying UniGAD's exact evaluation & thresholding protocol.
"""

import sys, os, torch, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.append(os.path.dirname(__file__))

# UniGAD's exact optimal threshold search protocol for Macro-F1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0.0, 0.0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = (probs > thres).astype(int)
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1, best_thre = mf1, thres
    return best_f1, best_thre

def evaluate(labels, probs):
    if torch.is_tensor(labels): labels = labels.cpu().numpy()
    if torch.is_tensor(probs):  probs = probs.cpu().numpy()
    macro_f1, best_thresh = get_best_f1(labels, probs)
    return {
        'AUROC': float(roc_auc_score(labels, probs)),
        'AUPRC': float(average_precision_score(labels, probs)),
        'Macro-F1': float(macro_f1),
        'Threshold': float(best_thresh)
    }
