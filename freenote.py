import matplotlib.pyplot as plt
import numpy as np
import os

alpha_gan = np.genfromtxt(os.path.abspath(r'C:\Users\rnt\Desktop\alpha-gan.csv'), delimiter=',')
alpha_ep_ori = [4, 5, 8, 10, 13, 15, 16, 19, 21, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 39, 40, 41, 43, 44, 51, 54,
            55, 57, 59, 61, 66, 69, 72, 74, 75, 76, 78, 79, 82, 83, 85, 86, 87, 90, 92, 93, 94, 97, 99, 103, 104, 105,
            106, 107, 109, 110, 111, 112, 113, 118, 119, 120, 122, 131]
ours_ep_ori = [84, 13, 74, 57, 75, 57, 21, 54, 11, 16, 57, 106, 100, 97, 47, 57, 63, 47, 37, 123, 24, 57, 123, 28, 78, 64,
           93, 100, 73, 75, 40, 73, 23, 35, 34, 54, 123, 35, 24, 14, 24, 73, 43, 41, 77, 106, 118, 21, 106, 49, 47, 43,
           63, 22, 118, 123, 74, 21, 23, 13, 63, 22, 17, 84, 63, 15]
alpha_ep_ori = np.asarray(alpha_ep_ori)
ours_ep_ori = np.asarray(ours_ep_ori)
alpha_ep = np.float_(alpha_ep_ori)/100+4
ours_ep = np.float_(ours_ep_ori)/100+4
raw_data = alpha_gan
#condition_idx = np.where((np.abs(raw_data[:,4]-1)<0.2) & ((raw_data[:,0]-1)<5) & ((raw_data[:,1]-1) < 5))
#data = raw_data[condition_idx][:-1]
condition_idx = np.where((np.abs(raw_data[:,4]-1)<0.5)&(raw_data[:,5]<20)&((raw_data[:,0]<10)&(raw_data[:,1]<10)))
# r test가 1에 가깝고 r sample이 10 이상이라 그래프를 방해하는 경우 제외. test 데이터 error률이 10%이상인 경우 제외

#data=np.column_stack([np.clip(data[:,0],a_min = None, a_max = 10), np.clip(data[:,1],a_min = None, a_max = 10),np.ones(120)])
#data=np.column_stack([np.abs(np.clip(data[:,0],a_min = None, a_max = 2)-np.ones(120)),np.clip(data[:,1],a_min = None, a_max = 10)])

data = raw_data[condition_idx][:,4:6]
#data = raw_data[condition_idx][:-2]

data = np.column_stack([data,alpha_ep[condition_idx],ours_ep[condition_idx],np.ones(len(data[:,0]))])
print(data)
plt.plot(data)
plt.legend(['R_test', 'R_sample', 'alpha-gan epoch', 'ours epoch', 'one'])
#plt.legend(['main real', 'ct real', 'main swap', 'ct swap', 'R_test','R_sample'])

plt.xticks(np.arange(len(data[:,0])),alpha_ep_ori[condition_idx])
plt.xlabel('valid idx')
plt.ylabel('error_rate')
plt.show()
print(1)

'''
alpha_gan_acc = alpha_gan[:,0]
ours_acc = alpha_gan[:,1]

result = []
for idx,i in enumerate(alpha_gan_acc):
    tmp = []
    for j in ours_acc:
        tmp.append(abs(i - j))
    max_idx = np.where(min(tmp) == np.asarray(tmp))[0]
    print('[%d]'%idx)
    if len(max_idx) == 1:
        if tmp[int(max_idx)] <= 1:
            if alpha_gan_acc[int(max_idx)] < 5:
                result.append([idx, int(max_idx), tmp[int(max_idx)], alpha_gan_acc[int(max_idx)]])
    elif len(max_idx) > 1:
        for k in range(len(max_idx)):
            if tmp[int(max_idx[k])] <= 1:
                if alpha_gan_acc[int(max_idx[k])] < 5:
                    result.append([idx, int(max_idx[k]), tmp[int(max_idx[k])], alpha_gan_acc[int(max_idx[k])]])
print(result)

print(list(np.int_(np.asarray(result)[:,0])))
print(list(np.int_(np.asarray(result)[:,1])))
'''