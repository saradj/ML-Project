# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as plt

from proj1_helpers import *
from implementations import *

def ridge_reg_cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    l_test = k_indices[k]
    l_train =  k_indices[~(np.arange(k_indices.shape[0])==k)] 
    l_train = l_train.reshape(-1)
    x_test = x[l_test]
    y_test = y[l_test]
    x_train = x[l_train] 
    y_train = y[l_train] 
    
    #print(x_train.shape)
    #x_train_d = build_poly(x_train, degree)
    #x_test_d = build_poly(x_test, degree)
   
    #print(x_train_d.shape)
  
    w, mse_loss = ridge_regression(y_train, x_train, lambda_)
    #print(w.shape)
    
    loss_tr = compute_loss(y_train, x_train, w)
    loss_te = compute_loss(y_test, x_test, w)
    #loss_tr = compute_rmse(y_train, x_train_d, w) 
    #loss_te = compute_rmse(y_test, x_test_d, w)  
    return loss_tr, loss_te

def reg_log_regression_cross_validation(y, x, k_indices, k, lambda_, initial_w, max_iters, gamma):
    """
    Cross validation k-fold for regularized logistic regression
    """
    # splitting the train data to take only the k'th set as test, rest is for training
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    test_set_idx = k_indices[k]
    train_set_idx = np.delete(k_indices, (k), axis=0).ravel()
    x_train = x[train_set_idx, :]
    x_test = x[test_set_idx, :]
    y = reg_log_regression_labels(y)
    y_train = y[train_set_idx]
    y_test = y[test_set_idx]

    x_train = preprocessing(x_train)
    x_test = preprocessing(x_test)
    weights, _ = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)
    # predict
    y_train_pred, _ = predict_labels_log_regression(y_train, weights, x_train)
    y_test_pred, _  = predict_labels_log_regression(y_test, weights, x_test)
    
    # compute accuracy
    acc_train = compute_accuracy(y_train_pred, y_train)
    acc_test = compute_accuracy(y_test_pred, y_test)

    return acc_train, acc_test
