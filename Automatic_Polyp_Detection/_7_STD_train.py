# laod 1st saliency map and calculaye it's mean
# all superpixel's value top 20% and bottom 25% are set to postive and negative, respectively
# selected superpixels are listed to samples.
# all samples's features are used to MKBoosting.
# train MKB. and make strong classifier
from Automatic_Polyp_Detection import superpixel as SUPERPIXEL
from sklearn.svm import SVR
import os
import LJY_utils
import numpy as np
from glob import glob
import random

result_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
superpixel_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/superpixels"
level_of_superpixels = 5
weight_for_the_combination = 0.4

LJY_utils.make_dir(result_path)


def custom_sign(r):
    result = (r > 0).astype(int)
    return result

def MKBerror(result_vector, label_vector, weights):
    #MKBerror = np.sum(weights * abs(result_vector) * custom_sign(np.multiply(label_vector*result_vector,-1)))/np.sum(weights*abs(result_vector))
    U = 0
    D = 0
    for i in range(0, len(label_vector)):
        U = weights[i] * custom_sign(np.multiply(label_vector[i]*result_vector[i],-1))
        D = weights[i]
    MKBerror = U / D
    if MKBerror == 0:
        MKBerror = 0.00000000000001

    return MKBerror


def MKB(weights, F, x):
    weights = weights/np.sum(weights)
    result = 0
    for i, fun in enumerate(F):
        # fun[0]: weight
        # fun[1]: SVM
        result = result + weights[i] * fun.predict(np.asarray(x).reshape(1,-1))
    return result


def MKBtrain(training_set,label_set):
    feature_list = ['color_hist', 'LM', 'LBP', 'HOG']
    num_of_features = len(feature_list)
    kernel_list = ['linear', 'rbf', 'sigmoid']
    num_of_kernels = len(kernel_list)
    iteration = 100
    Fweight = []
    FSVM = []
    # weights [features][kernels]
    weights = [1/len(label_set) for D in range(0, len(label_set))]
    SVMs = [[[] for x in range(0, num_of_kernels)] for y in range(0, num_of_features)]
    for i, feature in enumerate(feature_list):
        for j, kernel in enumerate(kernel_list):
            clf = SVR(kernel=kernel)
            clf.fit(training_set, label_set)
            SVMs[i][j] = clf
            print("[%d/%d] making classifier.." % (i, len(feature_list)))

    min_error = 100
    min_error_index = [0, 0]

    for i in range(0, iteration):
        print("[%d/%d] training.." % (i, iteration))
        for f, feature in enumerate(feature_list):
            for k, kernel in enumerate(kernel_list):
                prediction = SVMs[f][k].predict(training_set)

                error = MKBerror(prediction, label_set, weights)

                if error < min_error:
                        if error != 0 :
                            min_error = error
                            min_error_index = [f, k]
                            min_prediction = prediction
                        if error != 0 :
                            error = 0.0000000000000001
                            min_error = error
                            min_error_index = [f, k]
                            min_prediction = prediction

        regressor_weight = 1/2 * np.log((1-min_error)/min_error)
        if regressor_weight < 0:
            break
        Fweight.append(regressor_weight)
        FSVM.append(SVMs[min_error_index[0]][min_error_index[1]])
        '''
        for k, kernel in enumerate(kernel_list):
            for f, feature in enumerate(feature_list):
                weights[f][k] = weights[f][k] * np.power(regressor_weight, (1 - np.sign(abs(prediction - label_set)-threshold)))
        '''
        weights = weights / ( 2 * np.sqrt(min_error*(1-min_error))) *\
                  np.exp(label_set*min_prediction*-1*regressor_weight)
        weights = weights/np.sum(weights)

    return Fweight, FSVM

def make_training_set(root_path):

    pos_image_list = glob(root_path + "/pos_sample" + "/*")
    pos_image_list.sort()
    neg_image_list = glob(root_path + "/neg_sample" + "/*")
    neg_image_list.sort()

    training_set = []
    label_set = []

    for k, image_path in enumerate(pos_image_list):
        _superpixel = SUPERPIXEL.superpixel(image_path)
        training_set.append(sum([_superpixel.color_hist_feature, _superpixel.HOG_feature, _superpixel.LBP_feature,
                             _superpixel.LM_feature], []))
        label_set.append(1)
        print("[%d/%d] positive set making ..." % ((k + 1), len(pos_image_list)))

    for k, image_path in enumerate(neg_image_list):
        _superpixel = SUPERPIXEL.superpixel(image_path)
        training_set.append(sum([_superpixel.color_hist_feature, _superpixel.HOG_feature, _superpixel.LBP_feature,
                                 _superpixel.LM_feature], []))
        label_set.append(-1)
        print("[%d/%d] negative set making ..." % ((k + 1), len(neg_image_list)))
        # sum(2d-list,[]) = [[1,2],[3,4]] => [1,2,3,4]
    return training_set, label_set

training_set, label_set = make_training_set(superpixel_root_path)
sample_training_set = random.sample(training_set, int(len(training_set)/10))
sample_label_set = random.sample(label_set, int(len(label_set)/10))

weights, clfs = MKBtrain(sample_training_set, sample_label_set)
np.save(os.path.join(os.path.dirname(result_path), "MKB_weights.npy"), weights)
np.save(os.path.join(os.path.dirname(result_path), "MKB_clfs.npy"), clfs)
#result = MKB(weights, clfs, testing_set)





# ---------------------------------------------------------------------------------------------
# final
