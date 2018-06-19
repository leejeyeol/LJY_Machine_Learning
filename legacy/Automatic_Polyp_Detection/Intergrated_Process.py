import argparse
import os
import random
import time

import Sparse_Autoencoder_model as model
import cv2
import dataset_featureset_4 as datasets
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
from scipy.spatial import distance
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic
from torch.autograd import Variable

import LJY_utils
import LM_Filter
from legacy.Automatic_Polyp_Detection import superpixel, mask_converter


def MKB(weights, F, x):
    weights = weights/np.sum(weights)
    result = 0
    for i, fun in enumerate(F):
        # fun[0]: weight
        # fun[1]: SVM
        result = result + weights[i] * fun.predict(np.asarray(x).reshape(1,-1))
    return result


def min_max_refresh(data, minmax):
    if np.max(data)>minmax[1]:
        minmax[1] = np.max(data)
    if np.min(data) < minmax[0]:
        minmax[0] = np.min(data)
    return minmax


def bigger_abs(minmax):
    if abs(minmax[0]) > abs(minmax[1]):
        return minmax[0]
    else:
        return minmax[1]

is_train = True


# for calculate LM feature...
LM_filter_bank = LM_Filter.makeLMfilters()
np.random.seed(72170300)
'''
root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB"
image_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/Remove_Boundary"
superpixel_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/superpixels/"
feature_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/features"
saliency_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/saliency_map"
'''
root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/frames"
image_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/frames/normal_/sampled_normal"
superpixel_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/frames/normal_/superpixels/"
feature_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/frames/normal_/features"
saliency_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/frames/normal_/saliency_map"

LJY_utils.make_dir(saliency_save_path)
LJY_utils.make_dir(feature_save_path)
LJY_utils.make_dir(os.path.join(superpixel_save_path, "pos_sample"))
LJY_utils.make_dir(os.path.join(superpixel_save_path, "neg_sample"))
start_time = time.time()

# STD parameters =======================================================================================================
if not is_train:
    weights = np.load(os.path.join(root_path, "MKB_weights.npy"))
    clfs = np.load(os.path.join(root_path, "MKB_clfs.npy"))
    weight_for_the_combination = 0.4
# ======================================================================================================================

image_path_list = LJY_utils.get_file_paths(image_root_path, "/*.", ['png', 'PNG'])


for cnt, image_path in enumerate(image_path_list):
    print("===============================")
    start_time_per_1_image = time.time()
    start_time_sp = time.time()
    superpixel_index = 0
    min_max_hist = [0, 0]
    min_max_hog = [0, 0]
    min_max_lm = [0, 0]
    min_max_lbp = [0, 0]
    superpixels = []
    indeces_per_levels = [[], [], [], [], []]

    image = cv2.imread(image_path)
    image_number = int(LJY_utils.extract_filename_from_path(image_path))
    grayimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    boundary_mask = grayimage > 30
    grayimage = np.float32(grayimage)
    dst = cv2.cornerHarris(grayimage, 2, 3, 0.04)
    point_list = np.column_stack(np.where(dst > 0.01 * dst.max()))
    object_center = np.int32(np.mean(point_list, axis=0))

    # Threshold for an optimal value, it may vary depending on the image.

    LAB_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    _, hog_image = hog(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), block_norm='L2-Hys', visualise=True)
    LBP_feature = local_binary_pattern(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 8 * 3, 3, method='uniform')
    LM_features = []
    for filter_number in range(0, 48):
        # filtering
        LM_feature = cv2.filter2D(cv2.cvtColor(image, code=cv2.COLOR_RGB2GRAY), -1,
                                      kernel=LM_filter_bank[:, :, filter_number])
        LM_features.append(LM_feature)


    for superpixel_level in range(0, 5):
        #1~5 level segments...
        segments = slic(image, n_segments=50*(superpixel_level+1), compactness=50, convert2lab=True)
        for i in range(np.min(segments), np.max(segments)+1):
            # if image belongs to a segment and is not a boundary, the mask of the corresponding index is true.
            superpixel_mask = (segments[:] == i)
            # superpixel && boundary.
            intergrated_mask = np.logical_and(boundary_mask, superpixel_mask)
            # [weight, height] => [weight, height, channel]. for calculate with 3channel image.
            intergrated_mask = intergrated_mask[:, :, np.newaxis]

            # check there is no masked image in this level.
            if intergrated_mask.any():
                superpixels.append(superpixel.superpixel(superpixel_save_path, mask_converter.mask_to_list(intergrated_mask),
                                                         superpixel_index, image_number, superpixel_level, LM_features=LM_features,
                                                         LAB_image=LAB_image, hog_image=hog_image, LBP_feature=LBP_feature))

                min_max_hist = min_max_refresh(superpixels[superpixel_index].color_hist_feature, min_max_hist)
                min_max_hog = min_max_refresh(superpixels[superpixel_index].HOG_feature, min_max_hog)
                min_max_lm = min_max_refresh(superpixels[superpixel_index].LM_feature, min_max_lm)
                min_max_lbp = min_max_refresh(superpixels[superpixel_index].LBP_feature, min_max_lbp)
                #superpixels[superpixel_index].save_superpixel()
                #print(superpixel_index)
                #print("Create %d superpixel"%superpixel_index)
                indeces_per_levels[superpixel_level].append(superpixel_index)
                superpixel_index = superpixel_index + 1

            #print(i)
    tm,real_time = LJY_utils.time_visualizer(start_time_sp, time.time())

    print(real_time + " done superpixelize ")
# normalize ============================================================================================================
    start_time_sae = time.time()
    normal_hist = bigger_abs(min_max_hist)
    normal_hog = bigger_abs(min_max_hog)
    normal_lm = bigger_abs(min_max_lm)
    normal_lbp = bigger_abs(min_max_lbp)

    for sp in superpixels:
        if not sp.is_SAE_feature:
            sp.color_hist_feature = sp.color_hist_feature / normal_hist
            sp.HOG_feature = sp.HOG_feature / normal_hog
            sp.LM_feature = sp.LM_feature / normal_lm
            sp.LBP_feature = sp.LBP_feature / normal_lbp
            if is_train:
                features = [sp.color_hist_feature, sp.HOG_feature, sp.LM_feature, sp.LBP_feature]
                np.save(os.path.join(feature_save_path, "%08d_features" % (sp.index)), features)

# ======================================================================================================================
# Options
# ======================================================================================================================
    parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
    parser.add_argument('--net_hist', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/trained_networks/net_hist_epoch_4260.pth', help="path of networks.(to continue training)")
    parser.add_argument('--net_LM', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/trained_networks/net_LM_epoch_4260.pth', help="path of networks.(to continue training)")
    parser.add_argument('--net_LBP', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/trained_networks/net_LBP_epoch_4260.pth', help="path of networks.(to continue training)")
    parser.add_argument('--net_HOG', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/trained_networks/net_HOG_epoch_4260.pth', help="path of networks.(to continue training)")

    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
    parser.add_argument('--iteration', type=int, default=1, help='number of epochs to train for')

    # these options are saved for testing
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--model', type=str, default='SAE', help='Model name')
    parser.add_argument('--nz', type=int, default=10, help='number of input channel.')

    parser.add_argument('--seed', type=int, default=72170300 ,help='manual seed')


    options = parser.parse_args()
    #print(options)




# seed set  ============================================================================================================
    if options.seed is None:
        options.seed = random.randint(1, 10000)
#print("Random Seed: ", options.seed)
    random.seed(options.seed)
    torch.manual_seed(options.seed)

# cuda set  ============================================================================================================
    options.cuda = True
    if options.cuda:
        torch.cuda.manual_seed(options.seed)

    torch.backends.cudnn.benchmark = True
    cudnn.benchmark = True
    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


#=======================================================================================================================
# Data and Parameters
#=======================================================================================================================

    # todo make custom dataloader
    dataloader = torch.utils.data.DataLoader(datasets.featureset_4_online(superpixels),
        batch_size=options.batchSize, shuffle=False, num_workers=options.workers)

    ngpu = int(options.ngpu)
    nz = int(options.nz)

# Models ===============================================================================================================

    color_histogram_input_size = 610
    net_hist = model.SAE(ngpu, color_histogram_input_size)
    net_hist.apply(LJY_utils.weights_init)
    if options.net_hist != '':
        net_hist.load_state_dict(torch.load(options.net_hist))
    #print(net_hist)

    LM_input_size = 96
    net_LM = model.SAE(ngpu, LM_input_size)
    net_LM.apply(LJY_utils.weights_init)
    if options.net_LM != '':
        net_LM.load_state_dict(torch.load(options.net_LBP)) #todo chage option LBP <-> LM
    #print(net_LM)

    LBP_input_size = 255
    net_LBP = model.SAE(ngpu, LBP_input_size)
    net_LBP.apply(LJY_utils.weights_init)
    if options.net_LBP != '':
        net_LBP.load_state_dict(torch.load(options.net_LM))
    #print(net_LBP)

    HOG_input_size = 255
    net_HOG = model.SAE(ngpu, HOG_input_size)
    net_HOG.apply(LJY_utils.weights_init)
    if options.net_HOG != '':
        net_HOG.load_state_dict(torch.load(options.net_HOG))
    #print(net_HOG)

    # container generate
    input_hist = torch.FloatTensor(options.batchSize, color_histogram_input_size)
    input_LM = torch.FloatTensor(options.batchSize, LM_input_size)
    input_LBP = torch.FloatTensor(options.batchSize, LBP_input_size)
    input_HOG = torch.FloatTensor(options.batchSize, HOG_input_size)

    if options.cuda:
        net_hist.cuda()
        net_LM.cuda()
        net_LBP.cuda()
        net_HOG.cuda()
        input_hist = input_hist.cuda()
        input_LM = input_LM.cuda()
        input_LBP = input_LBP.cuda()
        input_HOG = input_HOG.cuda()
# make to variables ====================================================================================================
    input_hist = Variable(input_hist)
    input_LM = Variable(input_LM)
    input_LBP = Variable(input_LBP)
    input_HOG = Variable(input_HOG)
    # ======================================================================================================================
    # training start
    for epoch in range(options.iteration):
        for i, (data_hist, data_LM, data_LBP, data_HOG, sp_index) in enumerate(dataloader, 0):
            real_cpu_hist = data_hist
            batch_size_hist = real_cpu_hist.size(0)
            input_hist.data.resize_(real_cpu_hist.size()).copy_(real_cpu_hist)
            data_hist = Variable(data_hist).cuda()

            real_cpu_LM = data_LM
            batch_size_LM = real_cpu_LM.size(0)
            input_LM.data.resize_(real_cpu_LM.size()).copy_(real_cpu_LM)
            data_LM = Variable(data_LM).cuda()

            real_cpu_LBP = data_LBP
            batch_size_LBP = real_cpu_LBP.size(0)
            input_LBP.data.resize_(real_cpu_LBP.size()).copy_(real_cpu_LBP)
            data_LBP = Variable(data_LBP).cuda()


            real_cpu_HOG = data_HOG
            batch_size_HOG = real_cpu_HOG.size(0)
            input_HOG.data.resize_(real_cpu_HOG.size()).copy_(real_cpu_HOG)
            data_HOG = Variable(data_HOG).cuda()

            _, h_hist = net_hist(input_hist)
            _, h_LM = net_LM(input_LM)
            _, h_LBP = net_LBP(input_LBP)
            _, h_HOG = net_HOG(input_HOG)

            if not superpixels[sp_index.numpy()[0]].is_SAE_feature:
                superpixels[sp_index.numpy()[0]].set_SAE_feature(h_hist.data.tolist()[0], h_HOG.data.tolist()[0],
                                                                 h_LM.data.tolist()[0], h_LBP.data.tolist()[0])
    tm, real_time = LJY_utils.time_visualizer(start_time_sae, time.time())
    print(real_time + " done make SAE features")

# ======================================================================================================================
# calculate saliency stage
    start_time_wbu = time.time()
    WBU_saliency_map = np.zeros(grayimage.shape)
    STD_saliency_map = np.zeros(grayimage.shape)

    max_OB_value = 0
    superpixel_list_for_sampling = []
    for i_sp, sp in enumerate(superpixels):
        if not is_train:
            sp.saliency_value_STD = MKB(weights, clfs, sum([sp.color_hist_feature, sp.HOG_feature, sp.LBP_feature, sp.LM_feature],[]))

        sp.saliency_value_WBU_OB = np.exp(((-1 * (np.square(object_center[0] - sp.center[0])))/(2*1800) +
                                           (-1 * (np.square(object_center[1]-sp.center[1])))/(2*1800)))
        if max_OB_value < sp.saliency_value_WBU_OB:
            max_OB_value = sp.saliency_value_WBU_OB
        for i_per_level in indeces_per_levels[sp.superpixel_level]:
            if i_sp != i_per_level:
                sp.saliency_value_WBU_CB = sp.saliency_value_WBU_CB + \
                                           (distance.euclidean(sp.color_hist_feature, superpixels[i_per_level].color_hist_feature) +
                                            distance.euclidean(sp.HOG_feature, superpixels[i_per_level].HOG_feature) +
                                            distance.euclidean(sp.LM_feature, superpixels[i_per_level].LM_feature) +
                                            distance.euclidean(sp.LBP_feature, superpixels[i_per_level].LBP_feature))/\
                                           (1 + distance.euclidean(sp.center, superpixels[i_per_level].center))
    for i_sp, sp in enumerate(superpixels):
        sp.saliency_value_WBU_OB = sp.saliency_value_WBU_OB / max_OB_value
        sp.saliency_value_WBU = sp.saliency_value_WBU_OB * sp.saliency_value_WBU_CB
        if not is_train:
            superpixel_list_for_sampling.append([sp.saliency_value_WBU, i_sp])

        for i_coord in range(1, len(sp.mask)):
            WBU_saliency_map[sp.mask[i_coord][0]][sp.mask[i_coord][1]] = \
                WBU_saliency_map[sp.mask[i_coord][0]][sp.mask[i_coord][1]] + sp.saliency_value_WBU / 5
            if not is_train:
                STD_saliency_map[sp.mask[i_coord][0]][sp.mask[i_coord][1]] = \
                    STD_saliency_map[sp.mask[i_coord][0]][sp.mask[i_coord][1]] + sp.saliency_value_STD / 5
    if not is_train:
        STD_saliency_map = STD_saliency_map + np.min(STD_saliency_map)
        STD_saliency_map = STD_saliency_map / np.max(STD_saliency_map)
        total_saliency_map = weight_for_the_combination * WBU_saliency_map + \
                             (1 - weight_for_the_combination) * STD_saliency_map
    # visualize
    #LJY_visualize_tools.Test_Image(WBU_saliency_map, normalize=True)
    #LJY_visualize_tools.Test_Image(STD_saliency_map, normalize=True)
    #LJY_visualize_tools.Test_Image(total_saliency_map, normalize=True)

    # save file
    WBU_saliency_map = WBU_saliency_map - WBU_saliency_map.min()
    WBU_saliency_map = WBU_saliency_map / WBU_saliency_map.max()
    WBU_saliency_map = WBU_saliency_map * 255
    cv2.imwrite(os.path.join(saliency_save_path,"%03d_WBU_saliency.png"%cnt), WBU_saliency_map)

    tm, real_time = LJY_utils.time_visualizer(start_time_wbu, time.time())
    print(real_time + " done calculating WBU ")
    '''
    if is_train:
        superpixel_list_for_sampling.sort()
        positive = int(len(superpixel_list_for_sampling)*0.2)
        negative = int(len(superpixel_list_for_sampling)-len(superpixel_list_for_sampling)*0.25)

        for i_pos_sample in range(0, positive):
            superpixels[i_pos_sample].set_save_path(os.path.join(superpixels[i_pos_sample].save_path, "pos_sample"))
            superpixels[i_pos_sample].save_superpixel()

        for i_neg_sample in range(negative, len(superpixel_list_for_sampling)):
            superpixels[i_neg_sample].set_save_path(os.path.join(superpixels[i_neg_sample].save_path, "neg_sample"))
            superpixels[i_neg_sample].save_superpixel()
    '''

    tm, real_time = LJY_utils.time_visualizer(start_time_per_1_image, time.time())
    print(real_time + " done [%d/%d] image " % (cnt, len(image_path_list)))
tm, real_time = LJY_utils.time_visualizer(start_time, time.time())
print(real_time+" done program %d"%cnt)



