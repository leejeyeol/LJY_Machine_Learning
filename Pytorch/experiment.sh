#!/bin/bash
for i in {1..3}
do
   python3 /home/mlpa/data_ssd/workspace/github/LJY_Machine_Learning/Pytorch/GAN/VAE+GAN.py --preset alpha-gan --cuda --dataset CelebA --runfunc GAM --pretrainedEpoch $i
done


