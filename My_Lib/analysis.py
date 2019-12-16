# Evaluation
import glob as glob
import numpy as np
import os
TP = 0
FP = 0
FN = 0
TN = 0

Negative = sorted(glob.glob(os.path.join('/home/leejeyeol/Downloads/selfconsistency - task_2/Negative', '*.*')))
Positive = sorted(glob.glob(os.path.join('/home/leejeyeol/Downloads/selfconsistency - task_2/Positive', '*.*')))



for path in Positive:
    if int(os.path.basename(path).split('.')[0].split('_')[1]) <= 300:
        TP += 1
        os.rename(path,os.path.join('/home/leejeyeol/Downloads/selfconsistency - task_2/TP',os.path.basename(path)))
    else:  # False
        FP += 1
        os.rename(path,os.path.join('/home/leejeyeol/Downloads/selfconsistency - task_2/FP',os.path.basename(path)))

for path in Negative:
    if int(os.path.basename(path).split('.')[0].split('_')[1]) <= 300:
        FN += 1
        os.rename(path,os.path.join('/home/leejeyeol/Downloads/selfconsistency - task_2/FN',os.path.basename(path)))

    else:  # False
        TN += 1
        os.rename(path,os.path.join('/home/leejeyeol/Downloads/selfconsistency - task_2/TN',os.path.basename(path)))


print("TP : %d\t FN : %d\t FP : %d\t TN : %d\t" % (TP, FN, FP, TN))
print("Accuracy : %f \t Precision : %f \t Recall : %f" % (
    (TP + TN) / (TP + TN + FP + FN), TP / (TP + FP), TP / (FN + TP)))
