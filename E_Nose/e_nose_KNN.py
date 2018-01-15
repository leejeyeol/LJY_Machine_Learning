import os
import numpy as np
import glob
import scipy.io
import sklearn.neighbors
from sklearn.svm import SVC
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

feature_dir = "/media/leejeyeol/74B8D3C8B8D38750/Data/e_nose"
decay_list = ["40% loss", "30% loss", "20% loss", "10% loss"]
method_list = ["L1PCA+IRF", "L1PCA", "L2PCA", "dmg"]
rep_list = ["_rep(1)", "_rep(2)", "_rep(3)", "_rep(4)", "_rep(5)", "_rep(6)", "_rep(7)", "_rep(8)"]

SVM = True
# load data
feature_dic = dict()
for decay in decay_list:
    for method in method_list:
        for i_rep, rep in enumerate(rep_list):
            f_list = glob.glob(os.path.join(feature_dir, decay, method, "*%s.mat" % rep))
            f_list.sort()
            for i_f, f in enumerate(f_list):
                if int(i_f / 8) == 0:
                    feature_dic[decay, method, int(i_f % 8), i_rep, "test"] = scipy.io.loadmat(f)[
                        list(scipy.io.loadmat(f).keys())[-1]]
                if int(i_f / 8) == 1:
                    feature_dic[decay, method, int(i_f % 8), i_rep, "tr"] = scipy.io.loadmat(f)[
                        list(scipy.io.loadmat(f).keys())[-1]]


# classification and calc accuracy
IRF_accuracy = [[], [], [], []]
L1pca_accuracy = [[], [], [], []]
L2pca_accuracy = [[], [], [], []]
dmg_accuracy = [[], [], [], []]
for i_rep in range(8):
    for i_decay, decay in enumerate(decay_list):
        for i_fold in range(8):
            if SVM:
                IRF_clf = SVC(gamma=0.05, kernel='linear')
                L1pca_clf = SVC(gamma=0.05, kernel='linear')
                L2pca_clf = SVC(gamma=0.05, kernel='linear')
                dmg_clf = SVC(gamma=0.05, kernel='linear')
            else:
                IRF_clf = sklearn.neighbors.KNeighborsClassifier(1)
                L1pca_clf = sklearn.neighbors.KNeighborsClassifier(1)
                L2pca_clf = sklearn.neighbors.KNeighborsClassifier(1)
                dmg_clf = sklearn.neighbors.KNeighborsClassifier(1)



            IRF_clf.fit(feature_dic[decay, method_list[0], i_fold, i_rep, "tr"][:, :-1],
                        feature_dic[decay, method_list[0], i_fold, i_rep, "tr"][:, -1])
            L1pca_clf.fit(feature_dic[decay, method_list[1], i_fold, i_rep, "tr"][:, :-1],
                          feature_dic[decay, method_list[1], i_fold, i_rep, "tr"][:, -1])
            L2pca_clf.fit(feature_dic[decay, method_list[2], i_fold, i_rep, "tr"][:, :-1],
                          feature_dic[decay, method_list[2], i_fold, i_rep, "tr"][:, -1])
            dmg_clf.fit(feature_dic[decay, method_list[3], i_fold, i_rep, "tr"][:, :-1],
                          feature_dic[decay, method_list[3], i_fold, i_rep, "tr"][:, -1])

            IRF_z = IRF_clf.predict(feature_dic[decay, method_list[0], i_fold, i_rep, "test"][:, :-1])
            IRF_accuracy[i_decay].append(accuracy_score(feature_dic[decay, method_list[0], i_fold, i_rep, "test"][:, -1], IRF_z))
            #print("IRF+PCA [%d,%d] : %f %s" % (i_rep,i_fold, IRF_accuracy[i_fold], decay))

            L1pca_z = L1pca_clf.predict(feature_dic[decay, method_list[1], i_fold, i_rep, "test"][:, :-1])
            L1pca_accuracy[i_decay].append(accuracy_score(feature_dic[decay, method_list[1], i_fold, i_rep, "test"][:, -1], L1pca_z))
            #print("L1PCA   [%d,%d] : %f %s" % (i_rep,i_fold, L1pca_accuracy[i_fold], decay))

            L2pca_z = L2pca_clf.predict(feature_dic[decay, method_list[2], i_fold, i_rep, "test"][:, :-1])
            L2pca_accuracy[i_decay].append(accuracy_score(feature_dic[decay, method_list[2], i_fold, i_rep, "test"][:, -1], L2pca_z))
            #print("L2PCA   [%d,%d] : %f %s" % (i_rep,i_fold, L2pca_accuracy[i_fold],decay))

            dmg_z = dmg_clf.predict(feature_dic[decay, method_list[3], i_fold, i_rep, "test"][:, :-1])
            dmg_accuracy[i_decay].append(accuracy_score(feature_dic[decay, method_list[3], i_fold, i_rep, "test"][:, -1], dmg_z))
            #print("dmg     [%d,%d] : %f %s" % (i_rep,i_fold, dmg_accuracy[i_fold], decay))

print("===========================")
print("avearage IRF accuracy 40 percent loss : %f" % (sum(IRF_accuracy[0]) / len(IRF_accuracy[0])))
print("avearage IRF accuracy 30 percent loss : %f" % (sum(IRF_accuracy[1]) / len(IRF_accuracy[1])))
print("avearage IRF accuracy 20 percent loss : %f" % (sum(IRF_accuracy[2]) / len(IRF_accuracy[2])))
print("avearage IRF accuracy 10 percent loss : %f" % (sum(IRF_accuracy[3]) / len(IRF_accuracy[3])))
print("===========================")
print("avearage L1pca accuracy 40 percent loss : %f" % (sum(L1pca_accuracy[0])/len(L1pca_accuracy[0])))
print("avearage L1pca accuracy 30 percent loss : %f" % (sum(L1pca_accuracy[1])/len(L1pca_accuracy[1])))
print("avearage L1pca accuracy 20 percent loss : %f" % (sum(L1pca_accuracy[2])/len(L1pca_accuracy[2])))
print("avearage L1pca accuracy 10 percent loss : %f" % (sum(L1pca_accuracy[3])/len(L1pca_accuracy[3])))
print("===========================")
print("avearage L2pca accuracy 40 percent loss : %f" % (sum(L2pca_accuracy[0])/len(L2pca_accuracy[0])))
print("avearage L2pca accuracy 30 percent loss : %f" % (sum(L2pca_accuracy[1])/len(L2pca_accuracy[1])))
print("avearage L2pca accuracy 20 percent loss : %f" % (sum(L2pca_accuracy[2])/len(L2pca_accuracy[2])))
print("avearage L2pca accuracy 10 percent loss : %f" % (sum(L2pca_accuracy[3])/len(L2pca_accuracy[3])))
print("===========================")
print("avearage demaged accuracy 40 percent loss : %f" % (sum(dmg_accuracy[0])/len(dmg_accuracy[0])))
print("avearage demaged accuracy 30 percent loss : %f" % (sum(dmg_accuracy[1])/len(dmg_accuracy[1])))
print("avearage demaged accuracy 20 percent loss : %f" % (sum(dmg_accuracy[2])/len(dmg_accuracy[2])))
print("avearage demaged accuracy 10 percent loss : %f" % (sum(dmg_accuracy[3])/len(dmg_accuracy[3])))

# visualization
n_groups = 4
# create plot

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

x_dmg = (sum(dmg_accuracy[3]) / len(dmg_accuracy[3]), sum(dmg_accuracy[2]) / len(dmg_accuracy[2]),
         sum(dmg_accuracy[1]) / len(dmg_accuracy[1]), sum(dmg_accuracy[0]) / len(dmg_accuracy[0]))

x_L2 = (sum(L2pca_accuracy[3]) / len(L2pca_accuracy[3]), sum(L2pca_accuracy[2]) / len(L2pca_accuracy[2]),
         sum(L2pca_accuracy[1]) / len(L2pca_accuracy[1]), sum(L2pca_accuracy[0]) / len(L2pca_accuracy[0]))

x_L1 = (sum(L1pca_accuracy[3]) / len(L1pca_accuracy[3]), sum(L1pca_accuracy[2]) / len(L1pca_accuracy[2]),
         sum(L1pca_accuracy[1]) / len(L1pca_accuracy[1]), sum(L1pca_accuracy[0]) / len(L1pca_accuracy[0]))

x_L1_IRF = (sum(IRF_accuracy[3]) / len(IRF_accuracy[3]), sum(IRF_accuracy[2]) / len(IRF_accuracy[2]),
         sum(IRF_accuracy[1]) / len(IRF_accuracy[1]), sum(IRF_accuracy[0]) / len(IRF_accuracy[0]))

rects1 = ax.bar(index - bar_width, x_dmg, bar_width,
                 alpha=opacity,
                 color='b',
                 label='x_dmg')

rects2 = ax.bar(index, x_L2, bar_width,
                 alpha=opacity,
                 color='r',
                 label='x_L2')

rects3 = ax.bar(index + bar_width, x_L1, bar_width,
                 alpha=opacity,
                 color='g',
                 label='x_L1')

rects4 = ax.bar(index + 2 * bar_width, x_L1_IRF, bar_width,
                 alpha=opacity,
                 color='c',
                 label='x_L1_IRF')

plt.xlabel('Loss rate')
plt.ylabel('Classification rate (%)')
plt.xticks(index + bar_width, ('10%', '20%', '30%', '40%'))
plt.legend()
plt.tight_layout()
plt.show()


data_to_box_plot = [dmg_accuracy[3], L2pca_accuracy[3], L1pca_accuracy[3], IRF_accuracy[3],
                    dmg_accuracy[2], L2pca_accuracy[2], L1pca_accuracy[2], IRF_accuracy[2],
                    dmg_accuracy[1], L2pca_accuracy[1], L1pca_accuracy[1], IRF_accuracy[1],
                    dmg_accuracy[0], L2pca_accuracy[0], L1pca_accuracy[0], IRF_accuracy[0]]
plt.boxplot(data_to_box_plot, notch=False, patch_artist=False)
plt.xlabel('Loss rate')
plt.ylabel('Classification rate (%)')
plt.tight_layout()
plt.savefig('/home/leejeyeol/Experiments/e_nose/test.eps')
