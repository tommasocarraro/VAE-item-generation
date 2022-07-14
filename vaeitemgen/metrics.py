import numpy as np


def auc(pos_pred, neg_pred):
    """
    It computes the AUC measure as stated in the fashion recommendation paper.

    :param pos_pred: np.array of predictions for relevant items
    :param neg_pred: np.array of predictions for non-relevant items
    :return: the AUC measure of the given predictions
    """
    return np.count_nonzero(pos_pred > neg_pred) / pos_pred.shape[0]