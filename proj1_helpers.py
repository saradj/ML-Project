# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data txrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def compute_accuracy(y_pred, y):
    """Computes accuracy"""
    sum = 0
    for idx, y_val in enumerate(y):
        if y_val == y_pred[idx]:
            sum += 1

    return sum / len(y)

def build_poly(x, degree):

    basis = np.ones((x.shape[0], 1))
    for power in range(1, degree+1):
        newCol = np.power(x, power)
        basis = np.concatenate((basis, newCol), axis=1)

    return basis

def preprocessing(x_train, x_test):
    """
    Replace all the missing values with the median and stack the log transform of the positive columns
    """
    # Impute missing data
    #x_train = x_train[:, ~np.all(x_train[1:] == x_train[:-1], axis=0)]
    #x_test = x_test[:, ~np.all(x_test[1:] == x_test[:-1], axis=0)]

    x_train,cols = replace_empty(x_train)
    x_test,cols_test = replace_empty(x_test)

    positive_val_cols = [0, 1, 2, 3, 5, 7,8, 9, 10, 13, 16, 19, 21, 23, 26, 29]

    # Apply log transform to the data x=>log(1+x)
    x_train_log = np.log(1 + x_train[:, positive_val_cols])
    #x_train = np.delete(x_train, positive_val_cols, 1)
    x_train = np.hstack((x_train, x_train_log))

    x_test_log = np.log(1 + x_test[:, positive_val_cols])
    #x_test = np.delete(x_test, positive_val_cols, 1)
    x_test = np.hstack((x_test, x_test_log))
    #x_train = np.delete(x_train,cols, axis=1)  
    #x_test = np.delete(x_test,cols_test, axis=1)  

    x_train, mean_x_train, std_x_train = standardize(x_train)
    x_test, mean_x_test, std_x_test = standardize(x_test, mean_x_train, std_x_train)
    #x_train = normalize(x_train)
    #x_test = normalize(x_test)

    return x_train, x_test

def normalize(x):
    """Standardize the original data set."""
    min_x = np.min(x, axis=0)
   
    max_x = np.max(x, axis=0)
    mindiff = x-min_x
    diff = max_x-min_x

    x[:, diff > 0]  = mindiff[:, diff > 0]/diff[ diff > 0]
    return x

def replace_empty(tx):
    ''' replace outliers with median of the feature'''
    empty_cols = list()             #keeps all the columns to remove if they are filled only with -999 value
    for i in range(0,tx.shape[1]):
        col = [x for x in tx[:,i] if x!=-999]    #removing all the empty values
        if len(col) == 0:                         
            empty_cols.append(i)
        else :
            median = np.median(col)
            index = np.where(tx[:,i] == -999)
            tx[index,i] = median                #replace the outliers with median in that column
    return tx, empty_cols



def standardize(x, mean_x=None, std_x=None):
    """ standardize the dataset by subtracting the mean and deviding by std """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    #x = x - mean_x
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]

    return x, mean_x, std_x

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv fortx for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` txching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[np.array(shuffle_indices).astype(int)]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]