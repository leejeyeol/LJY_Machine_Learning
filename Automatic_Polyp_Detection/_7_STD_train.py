# laod 1st saliency map and calculaye it's mean
# all superpixel's value top 20% and bottom 25% are set to postive and negative, respectively
# selected superpixels are listed to samples.
# all samples's features are used to MKBoosting.
# train MKB. and make strong classifier
import Automatic_Polyp_Detection._2_Superpixelize_SLIC as SUPERPIXEL
from sklearn.svm import SVR
import os
import LJY_utils
import numpy as np
from glob import glob

result_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Results"
superpixel_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/superpixels"
level_of_superpixels = 5
weight_for_the_combination = 0.4

LJY_utils.make_dir(result_path)


def custom_sign(r):
    result = (r>0).astype(int)
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
    SVMs = [[[] for x in range(0,num_of_kernels)] for y in range(0,num_of_features)]
    for i, feature in enumerate(feature_list):
        for j, kernel in enumerate(kernel_list):
            clf = SVR(kernel=kernel)
            clf.fit(training_set, label_set)
            SVMs[i][j] = clf

    min_error = 100
    min_error_index = [0,0]

    for i in range(0, iteration):
        print("[%d/%d] training.."%(i,iteration))
        for f, feature in enumerate(feature_list):
            for k, kernel in enumerate(kernel_list):
                prediction = SVMs[f][k].predict(training_set)

                error = MKBerror(prediction, label_set, weights)

                if error < min_error :
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
        weights = weights / ( 2 * np.sqrt(min_error*(1-min_error))) * np.exp(label_set*min_prediction*-1*regressor_weight)
        weights = weights/np.sum(weights)

    return Fweight, FSVM

def make_training_set(root_path):

    image_list = glob(root_path + "/*/")
    image_list.sort()
    training_set = []
    label_set = []


    for k, image_path in enumerate(image_list):
        superpixel_list = LJY_utils.get_file_paths(image_path, "/*.", ['txt', 'TXT'])
        means = []
        for i in range(1, level_of_superpixels+1):
            means.append(np.int(np.load(os.path.join(image_path, "%d_level_WBU_saliency_mean.npy")%i)))
        for superpixel in superpixel_list:
            superpixel_level = int(os.path.basename(superpixel).split('.')[0].split('_')[0])
            _superpixel = SUPERPIXEL.superpixel(superpixel)
            if _superpixel.saliency_value_WBU > (0.8) * means[(superpixel_level-1)]:
                training_set.append(sum([_superpixel.color_hist_feature, _superpixel.HOG_feature, _superpixel.LBP_feature,
                                     _superpixel.LM_feature], []))
                label_set.append(1)
            elif _superpixel.saliency_value_WBU < (0.25) * means[(superpixel_level-1)]:
                training_set.append(sum([_superpixel.color_hist_feature, _superpixel.HOG_feature, _superpixel.LBP_feature,
                                     _superpixel.LM_feature], []))
                label_set.append(-1)
        # sum(2d-list,[]) = [[1,2],[3,4]] => [1,2,3,4]

        print("[%d/%d] training set making ..."%((k+1), len(image_list)))
    return training_set, label_set





training_set, label_set = make_training_set(superpixel_root_path)
weights, clfs = MKBtrain(training_set, label_set)
np.save(os.path.join(os.path.dirname(result_path),"MKB_weights.npy"),weights)
np.save(os.path.join(os.path.dirname(result_path),"MKB_clfs.npy"),clfs)
#result = MKB(weights, clfs, testing_set)





# ---------------------------------------------------------------------------------------------
# final
