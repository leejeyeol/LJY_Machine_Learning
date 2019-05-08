#!/bin/bash
for i in {0..225}
do
   python3 /home/mlpa/data_ssd/workspace/github/LJY_Machine_Learning/Pytorch/GAN/VAE+GAN.py --preset alpha-gan --cuda --dataset CelebA --batchSize 384 --runfunc GAM --pretrainedEpoch $i
done

for i in {0..253}
do
   python3 /home/mlpa/data_ssd/workspace/github/LJY_Machine_Learning/Pytorch/GAN/VAE+GAN.py --preset ours --cuda --dataset CelebA --batchSize 384 --runfunc GAM --pretrainedEpoch $i
done


