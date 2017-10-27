# laod 1st saliency map and calculaye it's mean
# all superpixel's value top 20% and bottom 25% are set to postive and negative, respectively
# selected superpixels are listed to samples.
# all samples's features are used to MKBoosting.
# train MKB. and make strong classifier
from sklearn.svm import SVR
import numpy as np

training_set = [['each feature'], ['label']]
feature_list = ['color_hist', 'LM', 'LBP', 'HOG']
num_of_features = 4
kernel_list = ['linear', 'rbf', 'sigmoid']
num_of_kernels = 3
iteration = 100
F = 0

def MKBerror(result_vector, label_vector, weights):
    threshold = (1.5/len(result_vector)*모든 샘플에 대해 각 샘플마다 (label - result_vector)의 l2 norm)
    error = 모든 샘플에 대해 각 샘플마다 weights * sign(abs(result - label)-threshold)
    return error


# weights [features][kernels]
weights = [[[1/num_of_features],[1/num_of_features],[1/num_of_features]],[[1/num_of_features],[1/num_of_features],[1/num_of_features]],[[1/num_of_features],[1/num_of_features],[1/num_of_features]],[[1/num_of_features],[1/num_of_features],[1/num_of_features]]]
SVMs = [[[],[],[]],[[],[],[]],[[],[],[]],[[],[],[]]]
for feature in feature_list:
    for kernel in kernel_list:
        clf = SVR(kernel=kernel)
        clf.fit(training_set[0], training_set[1])
        SVMs[feature][kernel] = clf

min_error = 100
min_error_index = [0,0]
for i in range(0, iteration):
    for feature in feature_list:
        for kernel in kernel_list:
            error = MKBerror(SVMs[feature][kernel].predict(testdata),label,weights[feature][kernel])
            if error < min_error :
                min_error = error
                min_error_index = [feature,kernel]
    if min_error < 0.5:
        F = F + (1/2) * np.log((1-min_error)/min_error)*SVMs[min_error_index[0],min_error_index[1]]
    else:
        break



