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
from PIL import Image
import numpy as np


def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        os.makedirs(path)
        print(path)
    return



root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/HMDB51"
save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/HMDB51/middle_block"
data_index = 0
video_type_list = glob.glob(os.path.join(root_path,'frames/*'))
for video_type_name in video_type_list:
    print(video_type_name)
    video_list = glob.glob(os.path.join(video_type_name, '*'))
    for video in video_list:
        print(video)
        subroot = os.path.join(root_path, 'frames')
        filename = os.path.basename(video)
        folder_path = os.path.join(subroot, os.path.basename(os.path.dirname(video)), filename)
        frame_list = glob.glob(os.path.join(folder_path,'*'))
        for i in range(len(frame_list)-2):
            pre_frame = np.array(Image.open(frame_list[i]), dtype='float')
            mid_frame = np.array(Image.open(frame_list[i+1]), dtype='float')
            nxt_frame = np.array(Image.open(frame_list[i+2]), dtype='float')
            np.save(os.path.join(save_path, "%09d_frames.npy" % data_index),
                    np.array([pre_frame, mid_frame, nxt_frame]))
            data_index = data_index + 1

        print(data_index)

print("debug")