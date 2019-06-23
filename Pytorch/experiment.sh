#!/bin/bash
for i in {0..3}
do
   python3 /home/mlpa/data_ssd/workspace/github/LJY_Machine_Learning/Pytorch/GAN/VAE+GAN.py --modelOutFolder /home/mlpa/data_4T/experiment_results/model/LJY/VAEGAN_cifar10_dcgan_1 --preset alpha-gan --cuda --dataset CIFAR10 --batchSize 64 --runfunc Generate --pretrainedEpoch $i
   python3 /home/mlpa/data_ssd/workspace/github/LJY_Machine_Learning/inception_score.py
done



