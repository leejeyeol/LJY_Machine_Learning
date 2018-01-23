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

anomaly_origin_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/frames/anomaly_/anomaly"
anomaly_saliency_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/frames/anomaly_/saliency_map"
anomaly_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/frames/anomaly_/intergrated_image"

normal_origin_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/frames/normal_/sampled_normal"
normal_saliency_path= "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/frames/normal_/saliency_map"
normal_save_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/frames/normal_/intergrated_image"

anomaly_origin = glob.glob(os.path.join(anomaly_origin_path, "*.*"))
anomaly_origin.sort()

anomaly_saliency = glob.glob(os.path.join(anomaly_saliency_path, "*.*"))
anomaly_saliency.sort()

normal_origin = glob.glob(os.path.join(normal_origin_path, "*.*"))
normal_origin.sort()

normal_saliency = glob.glob(os.path.join(normal_saliency_path, "*.*"))
normal_saliency.sort()

for i in range(len(anomaly_origin)):
    ori_img = cv2.imread(anomaly_origin[i])
    sal_img = cv2.imread(anomaly_saliency[i])
    result = np.hstack((ori_img, sal_img))
    cv2.imwrite(os.path.join(anomaly_save_path, "%03d.png" % i),result)

for i in range(len(normal_origin)):
    ori_img = cv2.imread(normal_origin[i])
    sal_img = cv2.imread(normal_saliency[i])
    result = np.hstack((ori_img, sal_img))
    cv2.imwrite(os.path.join(normal_save_path, "%03d.png" % i),result)
