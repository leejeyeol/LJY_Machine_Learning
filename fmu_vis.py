import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use(['makina-notebook'])

#train_log_file = r'C:\Users\rnt\Downloads\Score_log.txt'
train_log_file = r'D:\experiments\HANON\Score_log.txt'

f = open(train_log_file, 'r')
lines = f.read().splitlines()
f.close()

train_lst = []
eval_lst = []
train_tem_list = []
eval_tem_list = []
episode_tem_list = []
cop = 0
train_cop_mean_list = []
eval_cop_mean_list = []

reward = 0
train_reward_mean_list = []
eval_reward_mean_list = []

for l in lines:
    split_e = l.split()
    if l.startswith('Episode_number'):
        if split_e[6] == 'False':
            flag = 'train'
        else:
            flag = 'eval'
        target_tem = float(split_e[3][:-1])

    if l.startswith("Step number"):
        episode_tem_list.append([float(split_e[10][:-1])-target_tem, 0]) # current temperature, target
        #cop += np.clip(float(split_e[12]), 0, 20)

    if l.startswith('Episode score'):
        if flag == 'train':
            train_lst.append(float(split_e[2][:-1]))
            train_tem_list.append(episode_tem_list)
            episode_tem_list = []
            #train_cop_mean_list.append(cop/20)
            #cop = 0
        else:
            eval_lst.append(float(split_e[2][:-1]))
            eval_tem_list.append(episode_tem_list)
            episode_tem_list = []
            # eval_cop_mean_list.append(cop / 20)
            # cop = 0

# train_cop_move = [sum(train_cop_mean_list[i:i + 10]) / 10 for i in range(len(train_cop_mean_list) - 10)]
# train_cop_move = [0] * 10 + train_cop_move
#
# eval_cop_move = [sum(eval_cop_mean_list[i:i + 10]) / 10 for i in range(len(eval_cop_mean_list) - 10)]
# eval_cop_move = [0] * 10 + eval_cop_move


train_move = [sum(train_lst[i:i+10])/10 for i in range(len(train_lst)-10)]
train_move = [0]*10 + train_move
eval_move = [sum(eval_lst[i:i+10])/10 for i in range(len(eval_lst)-10)]
eval_move = [0]*10 + eval_move


#train_iter = len(train_tem_list)
train_iter = 880
for epi in range(train_iter):
    # cop = train_cop_mean_list[epi]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(np.asarray(train_tem_list)[epi], marker='o', alpha=0.7, markersize=5)
    ax[0].set_title('Episode : %d' % (epi + 1))
    ax[0].set_ylim(-20, 50)
    ax[0].set_xlim(0, 20)
    ax[0].set_xticks(np.arange(0, 20, step=5))
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Temperature')
    ax[0].legend(['temperature', 'zero'])

    # ax[1].plot(np.asarray(train_cop_move)[:epi], marker='o', alpha=0.7, markersize=2)
    # ax[1].set_ylim(0, 2.5)
    # ax[1].set_xlim(0, train_iter)
    # ax[1].set_xlabel('Episode')
    # ax[1].set_ylabel('COP')
    # ax[1].legend(['COP'])

    ax[2].plot(np.asarray(train_move)[:epi], marker='o', alpha=0.7, markersize=2)
    ax[2].set_ylim(0, 80)
    ax[2].set_xlim(0, train_iter)
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('Return')
    ax[2].legend(['Return'])

    #plt.show()
    plt.savefig(r'D:\experiments\HANON\figure_animation\train\fig_%05d'%(epi))
    plt.close()
    print('[%d/%d]'%(epi, train_iter))

#evel_iter = len(eval_tem_list)
evel_iter = 75
for epi in range(evel_iter):
    # cop = train_cop_mean_list[epi]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(np.asarray(eval_tem_list)[epi], marker='o', alpha=0.7,markersize = 5)
    ax[0].set_title('Episode : %d' % (epi + 1))
    ax[0].set_ylim(-20, 50)
    ax[0].set_xlim(0, 20)
    ax[0].set_xticks(np.arange(0, 20, step=5))
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Temperature')
    ax[0].legend(['temperature', 'zero'])

    # ax[1].plot(np.asarray(eval_cop_move)[:epi], marker='o', alpha=0.7, markersize=2)
    # ax[1].set_ylim(0, 2.5)
    # ax[1].set_xlim(0, evel_iter)
    # ax[1].set_xlabel('Episode')
    # ax[1].set_ylabel('COP')
    # ax[1].legend(['COP'])

    ax[2].plot(np.asarray(eval_move)[:epi], marker='o', alpha=0.7, markersize=2)
    ax[2].set_ylim(0, 80)
    ax[2].set_xlim(0, evel_iter)
    ax[2].set_xlabel('Episode')
    ax[2].set_ylabel('Return')
    ax[2].legend(['Return'])

    #plt.show()
    plt.savefig(r'D:\experiments\HANON\figure_animation\eval\fig_%05d'%(epi))
    plt.close()
    print('[%d/%d]'%(epi, evel_iter))

# imgs to video
# FFMPEG 설치 필요
os.system(r'ffmpeg -r 16  -i D:\experiments\HANON\figure_animation\train\fig_%05d.png  -vf scale="480:480"  -filter:v "setpts=PTS"  -vcodec libx264 -pix_fmt yuv420p -acodec libvo_aacenc -ab 128k -y D:\experiments\HANON\figure_animation\train_x2.mov')
os.system(r'ffmpeg -r 8  -i D:\experiments\HANON\figure_animation\train\fig_%05d.png  -vf scale="480:480"  -filter:v "setpts=PTS"  -vcodec libx264 -pix_fmt yuv420p -acodec libvo_aacenc -ab 128k -y D:\experiments\HANON\figure_animation\train.mov')
os.system(r'ffmpeg -r 8  -i D:\experiments\HANON\figure_animation\eval\fig_%05d.png  -vf scale="480:480"  -filter:v "setpts=PTS"  -vcodec libx264 -pix_fmt yuv420p -acodec libvo_aacenc -ab 128k -y D:\experiments\HANON\figure_animation\eval.mov')


