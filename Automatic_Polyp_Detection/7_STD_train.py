# laod 1st saliency map and calculaye it's mean
# all superpixel's value top 20% and bottom 25% are set to postive and negative, respectively
# selected superpixels are listed to samples.
# all samples's features are used to MKBoosting.
# train MKB. and make strong classifier
from sklearn.svm import SVR

training_set = [['each feature'], ['label']]
feature_list = ['color_hist', 'LM', 'LBP', 'HOG']
num_of_features = 4
kernel_list = ['linear', 'rbf', 'sigmoid']
num_of_kernels = 3
iteration = 100


# weights [features][kernels]
weights = [[[1/num_of_features],[1/num_of_features],[1/num_of_features]],[[1/num_of_features],[1/num_of_features],[1/num_of_features]],[[1/num_of_features],[1/num_of_features],[1/num_of_features]],[[1/num_of_features],[1/num_of_features],[1/num_of_features]]]

for feature in feature_list:
    for kernel in kernel_list:
        clf = SVR(kernel=kernel)
        clf.fit(training_set[0], training_set[1])
        clf.predict(test_set[0])