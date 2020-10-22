# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as plt

from proj1_helpers import *



def k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, method, **args):
    """
    Cross validation k-fold using the method passed as an argument
    """
  # splitting the train data to take only the k'th set as test, rest is for training
    test_set_idx = k_indices[k]
    train_set_idx = np.delete(k_indices, (k), axis=0).ravel()
    print("inside cv")
    x_train = x[train_set_idx, :]
    x_test = x[test_set_idx, :]
    y_train = y[train_set_idx]
    y_test = y[test_set_idx]

    x_train, x_test = preprocessing(x_train, x_test, True)

    weights, _ = method(y_train, x_train, **args)
    # predict
    y_train_pred = predict_labels(weights, x_train)
    y_test_pred = predict_labels(weights, x_test)
    # compute accuracy
    acc_train = compute_accuracy(y_train_pred, y_train)
    acc_test = compute_accuracy(y_test_pred, y_test)

    return acc_train, acc_test


