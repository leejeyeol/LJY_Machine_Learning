import matplotlib as plt
import numpy as np
import os

alpha_gan = np.genfromtxt(os.path.abspath(r'C:\Users\rnt\Desktop\alpha-gan.csv'), delimiter=',')
ours = np.genfromtxt(r'C:\Users\rnt\Desktop\ours.csv', delimiter=',')

data = alpha_gan
data=np.column_stack([data[:,0], data[:,2]])

plt.plot(data)
#plt.legend(['gradient of discriminator','gradient of decoder', 'gradient of generator'])
plt.legend(['gradient of discriminator', 'gradient of generator'])

plt.xlabel('Iteration')
plt.ylabel('Norm of Gradient')
plt.show()
print(1)


