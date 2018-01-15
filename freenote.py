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

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/boxplot"
data_path_list = glob.glob(os.path.join(root_path, "*.dat"))
data_path_list.sort()
print(data_path_list)

data_name = ["NLDA", "PCALDA", "PCALDA_RMS3", "PCALDA_RMS31_2", "PCALDA_RMS31_3", "PCALDA_RMS32_2", "PCALDA_RMS32_3"]
data_dic = {}
for name in data_name:
    data_dic[name] = []

for data_path in data_path_list:
    if os.path.basename(data_path).split('_')[0] == "correc":
        data_dic["NLDA"].append(np.loadtxt(data_path).flatten())
    elif os.path.basename(data_path).split('_')[2] == "mat8fold":
        data_dic["PCALDA"].append(np.loadtxt(data_path).flatten())
    elif os.path.basename(data_path).split('_')[-1] == "RMS3.dat":
        data_dic["PCALDA_RMS3"].append(np.loadtxt(data_path).flatten())
    elif os.path.basename(data_path).split('_')[-2] == "RMS31":
        if os.path.basename(data_path).split('_')[-1] == "(2).dat":
            data_dic["PCALDA_RMS31_2"].append(np.loadtxt(data_path).flatten())
        if os.path.basename(data_path).split('_')[-1] == "(3).dat":
            data_dic["PCALDA_RMS31_3"].append(np.loadtxt(data_path).flatten())
    elif os.path.basename(data_path).split('_')[-2] == "RMS32":
        if os.path.basename(data_path).split('_')[-1] == "(2).dat":
            data_dic["PCALDA_RMS32_2"].append(np.loadtxt(data_path).flatten())
        if os.path.basename(data_path).split('_')[-1] == "(3).dat":
            data_dic["PCALDA_RMS32_3"].append(np.loadtxt(data_path).flatten())

data_for_box_plot = []
for i_key, key in enumerate(list(data_dic.keys())):
    print(key)
    for i in range(5):
        data_for_box_plot.append(data_dic[key][i])
labels = ["5%", "10%", "15%", "20%", "25%"]*7
colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']*7
bp = plt.boxplot(data_for_box_plot, labels=labels, notch=False, patch_artist=True)


'''
fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(6, 6), sharey=True)

for i in range(7):
    axes[i].boxplot(data_for_box_plot[i], labels=labels, notch=False, patch_artist=True)
    axes[i].set_title(data_name[i], fontsize=10)
    
    for patch, color in zip(axes[i]['boxes'], colors):
        patch.set_facecolor(color)
    '''


for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)


plt.xlabel('Loss rate')
plt.ylabel('Classification rate (%)')
plt.tight_layout()
plt.show()

print("done")
