import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

data = np.genfromtxt(r'D:\experiments\dcgan_MNIST_GAN_result.csv', delimiter=',')
data = np.column_stack((data[:,2],data[:,0]))
plt.xlabel("Iteration")
plt.ylabel("Sum of absolute values of gradient")
plt.plot(data[0:200,:])
plt.legend(['Generator','Discriminator'])
plt.show()
