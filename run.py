# Useful starting lines
import numpy as np
from proj1_helpers import *
from cross_validation import *
from implementations import *

# Define seed for train/test random splitting
seed = 10

DATA_TRAIN_PATH = 'train.csv' 
DATA_TEST_PATH = 'test.csv' 

y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
degree = 2
tX, tX_test = preprocessing(tX, tX_test)
poly_basis = build_poly(tX, degree)
print("building poly")
poly_basisTest = build_poly(tX_test, 2)
print("building poly")

w, loss = least_squares(y, poly_basis)
print(loss)
y_pred = predict_labels(w, poly_basisTest)
print("predicted ", str((y_pred==-1).sum()), "-1s and ", str((y_pred==1).sum()), "1s")
create_csv_submission(ids_test, y_pred, "least_squares.csv")
""" cross validation
k_fold = 10
#gamma = [0.1, 0.6, 0.01, 0.001]
#lambda_ = [0.000001, 0.04, 0.01, 0.00001, 0.001]
gamma = [ 0.6, 0.7, 0.8, 0.9, 0.06]
lambda_ = [0.000001, 0.01, 0.00001, 0.001]
max_iters = 1000 # try with less iterations maybe 
from collections import defaultdict

# Split data in k-fold
k_indices = k_indices(y, k_fold, seed)

best_acc_test = {}
best_acc_train = {}
best_g = {}
best_lambda = {}
for k in range(k_fold):
    best_acc_train[k] = 0
    best_acc_test[k] = 0
    
    for g in gamma:
        for l in lambda_:
            acc_train, acc_test= cross_validation(y, tX, k_indices, k, reg_logistic_regression, lambda_=l, initial_w=None, max_iters=max_iters, gamma=g)
            if(acc_train>best_acc_train[k]):
                best_acc_train[k] = acc_train
            if(acc_test>best_acc_test[k]):
                best_acc_test[k] = acc_test
                best_lambda[k] = l
                best_g[k] = g

            print("%f %f %d - Training accuracy: %f / Test accuracy : %f" % (l, g,k,acc_train, acc_test))
print(best_acc_train)
print(best_acc_test)
print(best_g)
print(best_lambda)
"""

#cross_validation_visualization([0,1], best_acc_train, best_acc_test)

#print("\nAverage test accuracy: %f" % np.mean(accs_test))
#print("Variance test accuracy: %f" % np.var(accs_test))
#print("Min test accuracy: %f" % np.min(accs_test))
#print("Max test accuracy: %f" % np.max(accs_test))
#tX, tX_test= preprocessing(tX, tX_test)
tX, _ = replace_empty(tX)
tX_test, _ = replace_empty(tX_test)

w, loss = reg_logistic_regression(y, tX, 0.01, None, 1000, 0.7)
print(loss)
y_pred = predict_labels(w, tX_test)
print("predicted ", str((y_pred==-1).sum()), "-1s and ", str((y_pred==1).sum()), "1s")
create_csv_submission(ids_test, y_pred, "reg_log_reg.csv")
# 0.665	0.363 for 
# 0.724	0.641 reg logistic preprocessing and 0.01, 0.7 and 100
# 0.770	0.625 least squares no preprocessing 
# 0.774	0.632 least squares preprocessing
# 0.2952679109171201
# 0.29526791091712734 removing the cols to remove
# 0.29526791091712734
# 0.2952679109171201
# 0.2952679109171201
# 0.2952679109171201
# 0.2952679109171228 also normalization
# 0.2952679109171201
# 0.2951405937065225
# 0.2951405937065295
# 0.2944385857259786 now