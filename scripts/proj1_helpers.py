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

def partition(y, data, ids, jet):
    jet_col = 22
    jet_indx = []

    for r,row in enumerate(data):
        if row[jet_col]==jet:
            jet_indx.append(r)
    y_i, data_i, ids_i = y[jet_indx], data[jet_indx], ids[jet_indx]   
    return jet_indx, y_i, data_i, ids_i

def remove_zero_var(data):
    new_data = data.T
    to_delete = []
    for i,col in enumerate(data.T):   
        std = np.std(col, axis=0)
        if std==0:
            to_delete.append(i)
    new_data = np.delete(new_data, to_delete,0)      
    return new_data.T

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data txrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def predict_labels_log_regression(y, weights, data):
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    correct_percentage = np.sum(y_pred == y) / float(len(y_pred))
    print('Percentage of correct predictions is: %', correct_percentage * 100)
    return y_pred, correct_percentage

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

def compute_precision_recall(y_pred, y):
    """compute the precison and recall"""
    true_pos = 0
    for i,y_n in enumerate(y_pred):
        if y_n==y[i]:
            true_pos+=1
            
    prec = true_pos / len(y_pred)
    recall = true_pos / len(y)
    return prec, recall
    
def compute_F1(y_pred, y):
    p, r = compute_precision_recall(y_pred, y)
    f1 = (2*p*r)/(p+r)
    return f1
    
    
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

def preprocessing(x_train):
    """
    Replace all the missing values with the median and stack the log, sqrt, sin, cos and cross
    transform of the columns 
    """

    # Remove colinear columns
    #x_train = x_train[:, ~np.all(x_train[1:] == x_train[:-1], axis=0)]

    x_train, cols = replace_empty(x_train)

    positive_val_cols = np.all(x_train >= 0, axis=0) # only use the positive columns for the log and sqrt transform

    # Apply log transform to the data x=>log(1+x)
    x_train_log = np.log(1 + x_train[:, positive_val_cols])
    x_train_sqrt = np.sqrt(x_train[:, positive_val_cols])
    x_train_sin = np.sin(x_train)
    x_train_cos = np.cos(x_train)
    x_train_cross = []
    D = x_train.shape[1]
    
    for i in range(D):
        for j in range(D):
            if i!=j:
                x_train_cross.append(x_train[:,i]*x_train[:,j])
             
    x_train_cross = np.array(x_train_cross).T           
    x_train = np.hstack((x_train, x_train_log))
    x_train = np.hstack((x_train, x_train_sqrt))
    x_train = np.hstack((x_train, x_train_sin))
    x_train = np.hstack((x_train, x_train_cos))
    x_train = np.hstack((x_train, x_train_cross))

    x_train, mean_x_train, std_x_train = standardize(x_train)
    #x_train = remove_outliers(x_train) 
    #x_train = normalize(x_train)
    return x_train

def remove_outliers(tX):
    clean_data = []
    for f in tX.T:
        mean = np.mean(f, axis=0)
        std = np.std(f, axis=0)
        edge = std * 3
        lowerb = mean - edge
        upperb = mean + edge
        for i,x in enumerate(f):
            if x<lowerb:
                f[i]=lowerb
            elif x>upperb:
                f[i]=upperb
                
    return tX

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
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]

    return x, mean_x, std_x

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


def split_data(x, y, ids, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
 
    l = np.random.permutation(np.arange(len(y)))
    s = int(ratio*len(y))
    l_train = l[:s] 
    l_test = l[s:]
    x_train = x[l_train]  
    x_test = x[l_test]
    y_train = y[l_train]
    y_test = y[l_test]
    ids_train = ids[l_train]
    ids_test = ids[l_test]
    
    return x_train, y_train, ids_train, x_test, y_test, ids_test
    


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def compute_loss(y, tx, w):
    pred = tx.dot(w)
    pred[np.where(pred>=0.5)] = 1
    pred[np.where(pred<0.5)] = -1
    err = y - pred
    loss= 1/2 * np.mean(err ** 2)
    #print(loss)
    return loss

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


def log_reg_labels(y):
    y[y == -1] = 0
    return y

def predict_labels_log_regression(y, weights, data):
    y[y == 0] = -1
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    correct_percentage = np.sum(y_pred == y) / float(len(y_pred))
    print('Percentage of correct predictions is: %', correct_percentage * 100)
    return y_pred
