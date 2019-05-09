import matplotlib.pyplot as plt
import numpy as np
import os

alpha_gan = np.genfromtxt(os.path.abspath(r'C:\Users\rnt\Desktop\alpha-gan.csv'), delimiter=',')
ours = np.genfromtxt(r'C:\Users\rnt\Desktop\ours.csv', delimiter=',')

data = alpha_gan[0:120, 4:6]
data=np.column_stack([np.clip(data[:,0],a_min = None, a_max = 10), np.clip(data[:,1],a_min = None, a_max = 10),np.ones(120)])

plt.plot(data)
plt.legend(['R_test', 'R_sample'])
#plt.legend(['main real', 'ct real', 'main swap', 'ct swap', 'R_test','R_sample'])
plt.xlabel('epoch')
plt.ylabel('error_rate')
plt.show()
print(1)


