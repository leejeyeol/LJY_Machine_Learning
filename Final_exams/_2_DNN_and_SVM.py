import sklearn
import numpy as np

number_of_folds = 10
train_sets = []
test_sets = []
train_labels = []
test_labels = []

for i in range(number_of_folds):
    train_data = np.genfromtxt('cv%d_tr.csv' % (i + 1), delimiter=',')[1:, 1:]
    test_data = np.genfromtxt('cv%d_te.csv'%(i+1), delimiter=',')[1:, 1:]
    train_sets.append(train_data[:, 1:])
    test_sets.append(test_data[:, 1:])
    train_labels.append(train_data[:, 0])
    test_labels.append(test_data[:, 0])
    print('debug')
'''
1 output , 13 input data
10-fold
col 1 is useless. remove it
*Normalize
training DNN and SVM
test => result
* result unnormalize
compute MAE, RMSE, CVRMSE
compute mean, std of these.

'''