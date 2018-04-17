'''
import gym
env = gym.make('Breakout-ram-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
'''

# !/bin/python3

# !/bin/python3
import glob
import os
import cv2
import numpy as np
import subprocess as sp

def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        os.makedirs(path)
        print(path)
    return

FFMPEG_BIN = "ffmpeg"
target_rows = 64
target_cols = 64


root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/HMDB51"
video_type_list=glob.glob(os.path.join(root_path,'videos/*'))
for video_type_name in video_type_list:
    make_dir(os.path.join(os.path.dirname(os.path.dirname(video_type_name)), 'frames', os.path.basename(video_type_name)))
    video_list = glob.glob(os.path.join(video_type_name, '*'))
    for video in video_list:
        subroot = os.path.join(root_path, 'frames')
        filename = os.path.basename(video)
        folder_path = os.path.join(subroot, os.path.basename(os.path.dirname(video)), filename.split('.')[0])
        make_dir(folder_path)
        # run ffmpeg command
        print('\tFrom: ' + filename)
        command = [FFMPEG_BIN,
                   '-i', video,
                   '-s', str(target_rows) + 'x' + str(target_cols),  # [rows x cols]
                   '-pix_fmt', 'rgb24',
                   '-vf', 'fps=30',
                   os.path.join(folder_path, 'frame_%07d.png')]
        sp.call(command)  # call command
        print("Extraction is done")
print("debug")