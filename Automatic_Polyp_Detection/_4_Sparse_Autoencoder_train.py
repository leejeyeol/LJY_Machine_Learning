import argparse
import os
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
import torch.optim as optim
# import custom package

import LJY_utils
import Sparse_Autoencoder_model as model
import dataset_featureset_4 as datasets



#=======================================================================================================================
# Options
#=======================================================================================================================
parser = argparse.ArgumentParser()
# Options for path =====================================================================================================
parser.add_argument('--dataroot', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/features', help='path to dataset')
parser.add_argument('--net_hist', default='', help="path of networks.(to continue training)")
parser.add_argument('--net_LM', default='', help="path of networks.(to continue training)")
parser.add_argument('--net_LBP', default='', help="path of networks.(to continue training)")
parser.add_argument('--net_HOG', default='', help="path of networks.(to continue training)")
parser.add_argument('--outf', default='/media/leejeyeol/74B8D3C8B8D38750/Data/CVC-ClinicDB/trained_networks', help="folder to output images and model checkpoints")

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--display', default=False, help='display options. default:False. NOT IMPLEMENTED')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--iteration', type=int, default=1, help='number of epochs to train for')

# these options are saved for testing
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
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
dataloader = torch.utils.data.DataLoader(datasets.featureset_4(options.dataroot, type='train'),
    batch_size=options.batchSize, shuffle=True, num_workers=options.workers)

# normalize to -1~1
ngpu = int(options.ngpu)
nz = int(options.nz)

#=======================================================================================================================
# Models
#=======================================================================================================================


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

# criterion set
criterion = nn.MSELoss()
criterion_sparsity = nn.KLDivLoss()
sparsity_penalty = 0.5
sparsity = torch.FloatTensor([5])
sparsity = Variable(sparsity).cuda()


# setup optimizer   ====================================================================================================
# todo add betas=(0.5, 0.999),
optimizer_hist = optim.Adam(net_hist.parameters(), weight_decay=0.002, lr=0.02)
optimizer_LM = optim.Adam(net_LM.parameters(), weight_decay=0.002, lr=0.02)
optimizer_LBP = optim.Adam(net_LBP.parameters(), weight_decay=0.002, lr=0.02)
optimizer_HOG = optim.Adam(net_HOG.parameters(), weight_decay=0.002, lr=0.02)

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
    criterion.cuda()
    input_hist = input_hist.cuda()
    input_LM=input_LM.cuda()
    input_LBP=input_LBP.cuda()
    input_HOG=input_HOG.cuda()


# make to variables ====================================================================================================
input_hist = Variable(input_hist)
input_LM = Variable(input_LM)
input_LBP = Variable(input_LBP)
input_HOG = Variable(input_HOG)



# training start
print("Training Start!")
for epoch in range(options.iteration):
    for i, (data_hist, data_LM, data_LBP, data_HOG, _) in enumerate(dataloader, 0):
        ############################
        # (1) Update D network
        ###########################
        # train with real data  ========================================================================================
        optimizer_hist.zero_grad()
        optimizer_LM.zero_grad()
        optimizer_LBP.zero_grad()
        optimizer_HOG.zero_grad()

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

        output_hist, h_hist = net_hist(input_hist)
        output_LM, h_LM = net_LM(input_LM)
        output_LBP, h_LBP = net_LBP(input_LBP)
        output_HOG, h_HOG = net_HOG(input_HOG)


        err_hist = criterion(output_hist, data_hist)#+ sparsity_penalty * criterion_sparsity(torch.sum(h_hist)/batch_size_hist, sparsity)
        err_hist.backward()
        err_LM = criterion(output_LM, data_LM)#+ sparsity_penalty * criterion_sparsity(torch.sum(h_LM)/batch_size_LM, sparsity)
        err_LM.backward()
        err_LBP = criterion(output_LBP, data_LBP)#+ sparsity_penalty * criterion_sparsity(torch.sum(h_LBP)/batch_size_LBP, sparsity)
        err_LBP.backward()
        err_HOG = criterion(output_HOG, data_HOG)#+ sparsity_penalty * criterion_sparsity(torch.sum(h_HOG)/batch_size_HOG, sparsity)
        err_HOG.backward()

        optimizer_hist.step()
        optimizer_LM.step()
        optimizer_LBP.step()
        optimizer_HOG.step()


        #visualize
        print('[%d/%d][%d/%d] Loss_hist: %.4f Loss_LM: %.4f Loss_LBP: %.4f Loss_HOG: %.4f'
              % (epoch, options.iteration, i, len(dataloader),
                 err_hist.data[0], err_LM.data[0], err_LBP.data[0], err_HOG.data[0]))


    # do checkpointing
        if i%10 == 0 :
            torch.save(net_hist.state_dict(), '%s/net_hist_epoch_%d.pth' % (options.outf, i))
            torch.save(net_LM.state_dict(), '%s/net_LM_epoch_%d.pth' % (options.outf, i))
            torch.save(net_LBP.state_dict(), '%s/net_LBP_epoch_%d.pth' % (options.outf, i))
            torch.save(net_HOG.state_dict(), '%s/net_HOG_epoch_%d.pth' % (options.outf, i))




# Je Yeol. Lee \[T]/
# Jolly Co-operation