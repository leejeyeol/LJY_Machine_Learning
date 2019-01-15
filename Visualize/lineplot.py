import numpy as np
import matplotlib.pyplot as plt

our_MNIST = np.genfromtxt('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/Final_exp/OURS_MNIST_result.csv', delimiter=',')
our_MNIST_GAN = np.genfromtxt('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/Final_exp/OURS_MNIST_GAN_result.csv', delimiter=',')
base_MNIST = np.genfromtxt('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/Final_exp/Base_MNIST_result.csv', delimiter=',')
base_MNIST_GAN = np.genfromtxt('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/Final_exp/Base_MNIST_GAN_result.csv', delimiter=',')

our_CelebA = np.genfromtxt('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/Final_exp/OURS_CelebA_result.csv', delimiter=',')
our_CelebA_GAN = np.genfromtxt('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/Final_exp/OURS_CelebA_GAN_result.csv', delimiter=',')
base_CelebA = np.genfromtxt('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/Final_exp/Base_CelebA_result.csv', delimiter=',')
base_CelebA_GAN = np.genfromtxt('/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/Final_exp/Base_CelebA_GAN_result.csv', delimiter=',')

data = our_MNIST_GAN[1:10000]
data=np.column_stack([data[:,0], data[:,2]])

plt.plot(data)
#plt.legend(['gradient of discriminator','gradient of decoder', 'gradient of generator'])
plt.legend(['gradient of discriminator', 'gradient of generator'])

plt.xlabel('Iteration')
plt.ylabel('Norm of Gradient')
plt.show()
print(1)