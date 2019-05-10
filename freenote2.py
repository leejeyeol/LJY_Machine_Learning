import matplotlib.pyplot as plt
import numpy as np
import os

alpha_gan = np.genfromtxt(os.path.abspath(r'C:\Users\rnt\Desktop\alpha-gan.csv'), delimiter=',')

'''
raw_data = alpha_gan
condition_idx = np.where((np.abs(raw_data[:,4]-1)<0.2) & ((raw_data[:,0]-1)<5) & ((raw_data[:,1]-1) < 5))
data = raw_data[condition_idx][:-1]


#data=np.column_stack([np.clip(data[:,0],a_min = None, a_max = 10), np.clip(data[:,1],a_min = None, a_max = 10),np.ones(120)])
#data=np.column_stack([np.abs(np.clip(data[:,0],a_min = None, a_max = 2)-np.ones(120)),np.clip(data[:,1],a_min = None, a_max = 10)])

print(data)
plt.plot(data)
#plt.legend(['R_test', 'R_sample'])
plt.legend(['main real', 'ct real', 'main swap', 'ct swap', 'R_test','R_sample'])
plt.xticks(np.arange(len(condition_idx[0][:-1])),condition_idx[0][:-1])
plt.xlabel('epoch')
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
