import cv2
import os
import glob as glob
image_folder = '/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN'

os.system('ffmpeg -r 1 -i /media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/MG/gan_batch_1_%06d.png -vcodec mpeg4 -vf scale="480:480" -framerate 1 -filter:v "setpts=PTS/8" -y /media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/gan_only.mp4')
os.system('ffmpeg -r 1 -i /media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/MG/ours_batch_1_%06d.png -vcodec mpeg4 -vf scale="480:480" -framerate 1 -filter:v "setpts=PTS/8"   -y /media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/ours.mp4')
os.system('ffmpeg -r 1 -i /media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/MG/vae_batch_1_%06d.png -vcodec mpeg4 -vf scale="480:480" -framerate 1 -filter:v "setpts=PTS/8" -y /media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/vae_only.mp4')
