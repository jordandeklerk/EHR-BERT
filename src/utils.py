import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import numpy as np
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def multi_label_metric(y_gt, y_pred, y_prob):
    """
    Calculate multiple metrics for multi-label classification.
    :param y_gt: Ground truth labels (binary matrix).
    :param y_pred: Predicted labels (binary matrix).
    :param y_prob: Prediction probabilities.
    :return: Tuple of metrics including Jaccard, PR AUC, average precision, average recall, and average F1 score.
    """
    f1_macro = np.mean([f1_score(y_gt[i], y_pred[i], average='macro') for i in range(len(y_gt))])
    roc_auc_macro = np.mean([roc_auc_score(y_gt[i], y_prob[i], average='macro') for i in range(len(y_gt))])
    pr_auc_macro = np.mean([average_precision_score(y_gt[i], y_prob[i], average='macro') for i in range(len(y_gt))])

    jaccard_scores, precision_scores, recall_scores = [], [], []
    for i in range(y_gt.shape[0]):
        intersection = np.logical_and(y_gt[i], y_pred[i]).sum()
        union = np.logical_or(y_gt[i], y_pred[i]).sum()
        jaccard_scores.append(intersection / union if union != 0 else 0)
        precision_scores.append(intersection / y_pred[i].sum() if y_pred[i].sum() != 0 else 0)
        recall_scores.append(intersection / y_gt[i].sum() if y_gt[i].sum() != 0 else 0)

    f1_scores = [2 * p * r / (p + r) if (p + r) != 0 else 0 for p, r in zip(precision_scores, recall_scores)]

    return np.mean(jaccard_scores), pr_auc_macro, np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)

def metric_report(y_pred, y_true, threshold=0.5):
    y_prob = y_pred.copy()
    y_pred = np.where(y_pred > threshold, 1, 0)

    ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(y_true, y_pred, y_prob)
    acc_container = {'jaccard': ja, 'f1': avg_f1, 'prauc': prauc}

    for k, v in acc_container.items():
        logger.info(f'{k:10s} : {v:10.4f}')

    return acc_container

def t2n(x):
    """
    Convert a torch tensor to a NumPy array.
    :param x: Torch tensor.
    :return: NumPy array.
    """
    return x.detach().cpu().numpy()

def get_n_params(model):
    """
    Calculate the total number of parameters in a model.
    :param model: PyTorch model.
    :return: Total number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)