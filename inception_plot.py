import numpy as np
import matplotlib.pyplot as plt
import re
import os

dirs = [r'C:\Users\rnt\Desktop\exp_dcgan', r'C:\Users\rnt\Desktop\exp_ours', r'C:\Users\rnt\Desktop\exp_wgan_gp']

for dir in dirs:

    f = open(os.path.join(dir, 'inception_score_graph.txt'), 'r')
    lines = f.read().splitlines()
    f.close()
    inception_score = []
    for line in lines:
        inception_score.append(float(line.split()[2]))
    plt.plot(np.asarray(inception_score))
    #plt.xticks(np.arange(0, 80000, step=1000))
    plt.xlabel('Iteration')
    plt.ylabel('Inception Score')
    plt.legend(['DCGAN', 'VLGAN', 'WGAN_GP'])
plt.show()
plt.close()


print(1)

