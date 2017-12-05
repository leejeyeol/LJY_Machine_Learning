# load image and classifed by strong classifier
# set saliency value
# make saliency map

from sklearn.svm import SVR
import numpy as np

def MKB(weights, F, x):
    weights = weights/np.sum(weights)
    result = 0
    for i, fun in enumerate(F):
        # fun[0]: weight
        # fun[1]: SVM
        result = result + weights[i] * fun.predict(x)
    return result

training_set = [[1,2,1,3],[1,2,3,1],[1,2,23,1]]
label_set = [1.0,1.0,-1.0]
test_set = [[3,1,2,1],[1,2,33,2],[1,2,25,1]]


result = MKB(weights, clfs, test_set)
print(result)