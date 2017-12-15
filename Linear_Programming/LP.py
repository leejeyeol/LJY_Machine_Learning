import numpy as np

# tabula
subjectto = [[-1,1,1,-3,1,0,0,1],
             [-5,-1,-3,-8,0,1,0,55],
             [1,-2,-3,5,0,0,1,3],
             [4,1,5,3,0,0,0,0]]
subjectto=np.asarray(subjectto).astype('float')



for _ in range(100):
    pivot_x = np.argmax(subjectto, 0)[len(subjectto)-1]

    tem_1 = subjectto[0:subjectto.shape[0]-1,subjectto.shape[1]-1]
    tem_2 = subjectto[0:subjectto.shape[0]-1,pivot_x]
    tem = []
    for t in range(len(tem_2)):
        if tem_2[t] == 0:
            tem.append(100000)
        else:
            tem.append(tem_1[t]/tem_2[t])

    pivot_y = np.argmin(tem)
    pivot_number = subjectto[pivot_x][pivot_y]

    pivot = (subjectto[pivot_x,:]/pivot_number)

    for y in range(subjectto.shape[0]):
        if y == pivot_y:
            subjectto[y] = pivot
        else:
            subjectto[y] = subjectto[y] - pivot * subjectto[y, pivot_y]
    print(subjectto)
    print("\n")
    if all(subjectto[len(subjectto)-1,:] <= 0):
        break
print("solution : %f"%subjectto[subjectto.shape[0]-1][subjectto.shape[1]-1])


print("debug line")