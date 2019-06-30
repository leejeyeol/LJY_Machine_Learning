import numpy as np
import matplotlib.pyplot as plt

'''
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
'''


inception_score1 = np.genfromtxt(r'C:\Users\rnt\Desktop\alpha-gan_CIFAR10_inception_score.csv', delimiter=',')

inception_score2 = np.genfromtxt(r'C:\Users\rnt\Desktop\ours_CIFAR10_inception_score.csv', delimiter=',')

inception_score = np.genfromtxt(r'C:\Users\rnt\Desktop\result.csv', delimiter=',')

print(inception_score)
ours = inception_score[0:180,0]
alpha = inception_score[181:-1,0]
alpha=np.concatenate((alpha,np.zeros(90)))

data = np.column_stack((alpha,ours))
'''
data = np.column_stack((inception_score1[:,0],inception_score2[:,0]))
'''
plt.plot(data)

#plt.legend(['gradient of discriminator','gradient of decoder', 'gradient of generator'])
plt.legend(['alphagan', 'ours'])

plt.xlabel('epoch')
plt.ylabel('inception score')
plt.show()

print(1)