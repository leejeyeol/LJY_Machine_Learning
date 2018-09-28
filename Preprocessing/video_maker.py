import cv2
import os
import glob as glob
image_folder = '/home/leejeyeol/Git/LJY_Machine_Learning/Pytorch/GAN/pretrained_model/MG_result'

os.system('ffmpeg -r 1 -i /home/leejeyeol/Git/LJY_Machine_Learning/Pytorch/GAN/pretrained_model/MG_result/ours_%06d.png -vcodec mpeg4 -vf scale="480:480" -framerate 1 -filter:v "setpts=PTS/8" -y /home/leejeyeol/Git/LJY_Machine_Learning/Pytorch/GAN/pretrained_model/ours_video.mp4')
os.system('ffmpeg -r 1 -i /home/leejeyeol/Git/LJY_Machine_Learning/Pytorch/GAN/pretrained_model/MG_result/original_%06d.png -vcodec mpeg4 -vf scale="480:480" -framerate 1 -filter:v "setpts=PTS/8"   -y /home/leejeyeol/Git/LJY_Machine_Learning/Pytorch/GAN/pretrained_model/original_video.mp4')
os.system('ffmpeg -r 1 -i /home/leejeyeol/Git/LJY_Machine_Learning/Pytorch/GAN/pretrained_model/MG_result/ours_aae_%06d.png -vcodec mpeg4 -vf scale="480:480" -framerate 1 -filter:v "setpts=PTS/8" -y /home/leejeyeol/Git/LJY_Machine_Learning/Pytorch/GAN/pretrained_model/ours_aae_video.mp4')
os.system('ffmpeg -r 1 -i /home/leejeyeol/Git/LJY_Machine_Learning/Pytorch/GAN/pretrained_model/MG_result/powerful_vae_%06d.png -vcodec mpeg4 -vf scale="480:480" -framerate 1 -filter:v "setpts=PTS/8" -y /home/leejeyeol/Git/LJY_Machine_Learning/Pytorch/GAN/pretrained_model/powerful_vae_video.mp4')
