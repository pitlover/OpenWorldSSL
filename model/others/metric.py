import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


def OpenSSLMetric(targets, preds, labeled_num):
    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask

    seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
    all_acc = cluster_acc(preds, targets)

    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])

    return seen_acc, unseen_acc, all_acc, unseen_nmi


def cluster_acc(y_pred, y_true):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size


def accuracy(output, target):
    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res
