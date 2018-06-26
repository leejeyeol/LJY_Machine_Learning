import numpy as np
import os

data_path = "/home/leejeyeol/Downloads/ml-20m/ratings.csv"
save_path = "/home/leejeyeol/Downloads/ml-20m"

print("now loading...")
A = np.genfromtxt(data_path, delimiter=',')
A = A[:, 0:3][1:]
item_start = 1
# item_end = int(max(A[:, 1])) MemoryError
item_end = int(1011)
user_start = 1
user_end = int(A[-1][0])

user_item_matrix = np.zeros([user_end, item_end])
for i in range(A.shape[0]):
    print("[%d/%d]making user item matrix.." % (i, A.shape[0]))
    user = A[i][0]
    item = A[i][1]
    rating = A[i][2]
    if item <= item_end:
        user_item_matrix[int(user-1)][int(item-1)] = rating

nonzero_row_indices = [i for i in range(user_item_matrix.shape[0]) if not np.allclose(user_item_matrix[i, :], 0)]
user_item_matrix_new = user_item_matrix[nonzero_row_indices, :]

np.save(os.path.join(save_path, "user_item_matrix_ml-20m.npy"), user_item_matrix_new)
print("done")
