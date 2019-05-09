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
print(1

'''
alpha_gan_acc = alpha_gan[:,0]
ours_acc = alpha_gan[:,1]


for i in alpha_gan_acc:
    for j in ours_acc:
        abs(i - j)

