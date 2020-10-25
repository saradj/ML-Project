import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = 'train.csv' # train data path  
y, raw_tX, ids = load_csv_data(DATA_TRAIN_PATH)
y = np.array(y)
raw_tX, cols = replace_empty(raw_tX)

jets = [0, 1, 2, 3]
degree = 2
lambda_ = 0.000774263682681127 # optimal value found by cross validation
jet_idxs = [None] * len(jets)
data = [None] * len(jets)
y_ = [None] * len(jets)
split_ids = [None] * len(jets)
tX_ = [None] * len(jets)
weights_ =[None] * len(jets)
loss_ = [None] * len(jets)
#------------------------------------------------TRAINING------------------------------------------------
for jet_nb in jets:
    jet_idxs[jet_nb], y_[jet_nb], data[jet_nb], split_ids[jet_nb] = partition(y, raw_tX, ids, jet_nb)
    tX_[jet_nb] = preprocessing(data[jet_nb])
    tX_[jet_nb] = build_poly(tX_[jet_nb], degree)
    weights_[jet_nb] = ridge_regression(y_[jet_nb], tX_[jet_nb], lambda_)
    loss_[jet_nb] = compute_loss(y_[jet_nb], tX_[jet_nb], weights_[jet_nb])
    print(loss_[jet_nb])


DATA_TEST_PATH = 'test.csv' # test data path
y_test, x_test, ids_test = load_csv_data(DATA_TEST_PATH)
x_test,_ = replace_empty(x_test)
y_pred_ = [None] * len(jets)

#------------------------------------------------PREDICTING-----------------------------------------------
for jet_nb in jets:
    jet_idxs[jet_nb], y_[jet_nb], data[jet_nb], split_ids[jet_nb] = partition(y_test, x_test, ids_test, jet_nb)
    tX_[jet_nb] = preprocessing(data[jet_nb])
    tX_[jet_nb] = build_poly(tX_[jet_nb], degree)
    y_pred_[jet_nb] = predict_labels(weights_[jet_nb], tX_[jet_nb])

ids_all = []
y_pred_all = []
#------------------------------------------------MERGING-----------------------------------------------
for id in split_ids:
    ids_all =  np.concatenate((ids_all, id))
for y in y_pred_:
    y_pred_all = np.concatenate((y_pred_all, y))

sorted_indx = np.argsort(ids_all)

OUTPUT_PATH = 'ridge.csv' 
ids_all = ids_all[sorted_indx]
y_pred_all = y_pred_all[sorted_indx]


create_csv_submission(ids_all, y_pred_all, OUTPUT_PATH)
