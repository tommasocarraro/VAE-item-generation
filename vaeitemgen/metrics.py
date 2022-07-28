import numpy as np


def auc(pos_pred, neg_pred):
    """
    It computes the AUC measure as stated in the fashion recommendation paper.

    :param pos_pred: np.array of predictions for relevant items
    :param neg_pred: np.array of predictions for non-relevant items. During validation we have 100 negative predictions
    for each positive prediction
    :return: the AUC measure of the given predictions
    """
    return np.mean(np.mean(pos_pred[:, np.newaxis] > neg_pred, axis=1))
