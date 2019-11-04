import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

data1 = np.genfromtxt(r'D:\experiments\dcgan_MNIST_GAN_result.csv', delimiter=',')
data2 = np.genfromtxt(r'D:\experiments\ours_MNIST_result.csv', delimiter=',')
data3 = np.genfromtxt(r'D:\experiments\dcgan_CelebA_GAN_result.csv', delimiter=',')
data4 = np.genfromtxt(r'D:\experiments\ours_CelebA_result.csv', delimiter=',')
data1 = np.column_stack((data1[:,2],data1[:,0]))
data2 = np.column_stack((data2[:,2],data2[:,0],data2[:,1]))
data3 = np.column_stack((data3[:,2],data3[:,0]))
data4 = np.column_stack((data4[:,2],data4[:,0],data4[:,1]))
fig, ax = plt.subplots(2,2,figsize=(6,6))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Iteration")
plt.ylabel("Sum of absolute values of gradient")


ax[0,0].plot(data1[0:200,:])
ax[0,0].set_title('(a)')
ax[0,0].legend(('Generator','Discriminator'),loc="upper right")
ax[0,1].plot(data2[0:200,:])
ax[0,1].set_title('(b)')
ax[0,1].legend(('Generator','Discriminator','Encoder'),loc="upper right")
ax[1,0].plot(data3)
ax[1,0].set_title('(c)')
ax[1,0].legend(('Generator','Discriminator'),loc="upper right")
ax[1,1].plot(data4)
ax[1,1].set_title('(d)')
ax[1,1].legend(('Generator','Discriminator','Encoder'),loc="upper right")


fig.tight_layout()
plt.show()
print(1)