# load features and test. extract hidden representation
# save hidden representations to superpixel structure.
# 3 hours

import argparse
import glob
import os
import random

import Sparse_Autoencoder_model as model
import dataset_featureset_4 as datasets
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable

import LJY_utils
from legacy.Automatic_Polyp_Detection import superpixel as SUPERPIXEL

# import custom package



#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/features_copy', help='path to dataset')
parser.add_argument('--net_hist', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/trained_networks/net_hist_epoch_4260.pth', help="path of networks.(to continue training)")
parser.add_argument('--net_LM', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/trained_networks/net_LM_epoch_4260.pth', help="path of networks.(to continue training)")
parser.add_argument('--net_LBP', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/trained_networks/net_LBP_epoch_4260.pth', help="path of networks.(to continue training)")
parser.add_argument('--net_HOG', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/trained_networks/net_HOG_epoch_4260.pth', help="path of networks.(to continue training)")
parser.add_argument('--outf', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/trained_networks', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--model', type=str, default='SAE', help='Model name')
parser.add_argument('--nz', type=int, default=10, help='number of input channel.')
parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam.')


parser.add_argument('--seed', type=int, help='manual seed')


options = parser.parse_args()
print(options)



# save directory make   ================================================================================================
try:
    os.makedirs(options.outf)
except OSError:
    pass

# seed set  ============================================================================================================
if options.seed is None:
    options.seed = random.randint(1, 10000)
print("Random Seed: ", options.seed)
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

# MNIST call and load   ================================================================================================

# todo make custom dataloader
dataloader = torch.utils.data.DataLoader(datasets.featureset_4(options.dataroot, type='test'),
    batch_size=options.batchSize, shuffle=False, num_workers=options.workers)

# normalize to -1~1
ngpu = int(options.ngpu)
nz = int(options.nz)

#=======================================================================================================================
# Models
#=======================================================================================================================
superpixel_root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/superpixels"
superpixel_image_list = glob.glob(superpixel_root_path+"/*/")
superpixel_image_list.sort()

superpixel_path_list=[]

for i, superpixel_image in enumerate(superpixel_image_list):
    superpixel_list = LJY_utils.get_file_paths(superpixel_image, "/*.", ['txt', 'TXT'])
    superpixel_path_list.append(superpixel_list)


# Generator ============================================================================================================
# todo get size automatically
color_histogram_input_size = 610
net_hist = model.SAE(ngpu, color_histogram_input_size)
net_hist.apply(LJY_utils.weights_init)
if options.net_hist != '':
    net_hist.load_state_dict(torch.load(options.net_hist))
print(net_hist)

# todo get size automatically
LM_input_size = 255
net_LM = model.SAE(ngpu, LM_input_size)
net_LM.apply(LJY_utils.weights_init)
if options.net_LM != '':
    net_LM.load_state_dict(torch.load(options.net_LM))
print(net_LM)

# todo get size automatically
LBP_input_size = 96
net_LBP = model.SAE(ngpu, LBP_input_size)
net_LBP.apply(LJY_utils.weights_init)
if options.net_LBP != '':
    net_LBP.load_state_dict(torch.load(options.net_LBP))
print(net_LBP)

# todo get size automatically
HOG_input_size = 255
net_HOG = model.SAE(ngpu, HOG_input_size)
net_HOG.apply(LJY_utils.weights_init)
if options.net_HOG != '':
    net_HOG.load_state_dict(torch.load(options.net_HOG))
print(net_HOG)


#=======================================================================================================================
# Training
#=======================================================================================================================



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
    input_LM=input_LM.cuda()
    input_LBP=input_LBP.cuda()
    input_HOG=input_HOG.cuda()


# make to variables ====================================================================================================
input_hist = Variable(input_hist)
input_LM = Variable(input_LM)
input_LBP = Variable(input_LBP)
input_HOG = Variable(input_HOG)


error_case = []
# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (data_hist, data_LM, data_LBP, data_HOG, feature_path) in enumerate(dataloader, 0):

        if data_hist.shape[1] == 610 and data_LM.shape[1] == 255 and data_LBP.shape[1] == 96 and data_HOG.shape[1] == 255 :
            ############################
            # (1) Update D network
            ###########################
            # train with real data  ========================================================================================
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


            print(feature_path)

            # todo extract superpixel number
            print("[%d][%d][%d]"%(int(os.path.basename(feature_path[0]).split('_')[0]), int(os.path.basename(feature_path[0]).split('_')[1]), len(superpixel_path_list[int(os.path.basename(feature_path[0]).split('_')[0])])))
            superpixel_path = superpixel_path_list[int(os.path.basename(feature_path[0]).split('_')[0])][int(os.path.basename(feature_path[0]).split('_')[1])]
            _superpixel = SUPERPIXEL.superpixel(superpixel_path)
            if not _superpixel.is_SAE_feature:
                _superpixel.set_SAE_feature(h_hist.data.tolist()[0], h_HOG.data.tolist()[0], h_LM.data.tolist()[0], h_LBP.data.tolist()[0])
                _superpixel.save_superpixel()
            #visualize
            print('[%d/%d][%d/%d]'
                  % (epoch, options.iteration, i, len(dataloader)))
        else:
            error_case.append(i)
print(error_case)



# Je Yeol. Lee \[T]/
# Jolly Co-operation