import numpy as np
import os
import random

num_of_dataset = 100

save_path = "/home/leejeyeol/Git/LJY_Machine_Learning/EM_RL/generated_data"
for dataset in range(num_of_dataset):

    data = []
    num_of_clusters = random.randint(2, 10)
    modal_informations = [num_of_clusters]
    for i_cluster in range(num_of_clusters):
        mean = [random.randint(0, 100), random.randint(0, 100)]
        cov = [[random.randint(0, 100), 0], [0, random.randint(0, 100)]]
        num_of_data = random.randint(500, 1000)
        modal_informations.append([i_cluster, mean, cov])
        x, y = np.random.multivariate_normal(mean, cov, num_of_data).T
        for i_data in range(num_of_data):
            data.append([[x[i_data], y[i_data]], i_cluster])
    #data = [label,[x,y]]
    #random.shuffle(data)
    data.insert(0, modal_informations)
    np.save(os.path.join(save_path, "%03d_data.npy" % (dataset)), data)
print("aaa")