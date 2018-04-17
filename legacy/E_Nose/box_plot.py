import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

root_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/boxplot2"
data_path_list = glob.glob(os.path.join(root_path, "*.dat"))
data_path_list.sort()
print(data_path_list)

data_name = ["PCALDA", "PCALDA_7", "NLDA", "PCALDA_RMS3", "PCALDA_RMS32"]
data_dic = {}
for name in data_name:
    data_dic[name] = []

for data_path in data_path_list:
    if os.path.basename(data_path).split('_')[0] == "correc":
        data_dic["NLDA"].append(np.loadtxt(data_path).flatten())
    elif os.path.basename(data_path).split('_')[0] == "Correct":
        data_dic["PCALDA_7"].append(np.loadtxt(data_path).flatten())
    elif os.path.basename(data_path).split('_')[2] == "mat8fold":
        data_dic["PCALDA"].append(np.loadtxt(data_path).flatten())
    elif os.path.basename(data_path).split('_')[-1] == "RMS3.dat":
        data_dic["PCALDA_RMS3"].append(np.loadtxt(data_path).flatten())
    elif os.path.basename(data_path).split('_')[-5] == "RMS32":
        data_dic["PCALDA_RMS32"].append(np.loadtxt(data_path).flatten())

data_for_box_plot = []
for i in range(4):
    for i_key, key in enumerate(list(data_dic.keys())):
        print(key)
        data_for_box_plot.append(data_dic[key][i])
labels = ["5%", "10%", "15%", "20%"] * 5
colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'red'] * 5


bp = plt.boxplot(data_for_box_plot, labels=labels, notch=False, patch_artist=True, showmeans=True)

'''
fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(6, 6), sharey=True)

for i in range(7):
    axes[i].boxplot(data_for_box_plot[i], labels=labels, notch=False, patch_artist=True)
    axes[i].set_title(data_name[i], fontsize=10)

    for patch, color in zip(axes[i]['boxes'], colors):
        patch.set_facecolor(color)
    '''

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

tickrate = np.arange(4)*5
width = 3
plt.xticks(tickrate+width)
plt.xlabel('Loss rate')
plt.ylabel('Classification rate (%)')
plt.tight_layout()
#['cyan', 'lightblue', 'lightgreen', 'tan', 'red']
legend_patch_1 = mpatches.Patch(color='cyan', label="$y^{dmg}$")
legend_patch_2 = mpatches.Patch(color='lightblue', label="$y^{re}_{L2}$")
legend_patch_3 = mpatches.Patch(color='lightgreen', label="$y^{FF}$")
legend_patch_4 = mpatches.Patch(color='tan', label="$y^{re}_{L1}$")
legend_patch_5 = mpatches.Patch(color='red', label="$y^{NLDA}$")


plt.legend(handles=[legend_patch_1, legend_patch_2, legend_patch_3, legend_patch_4, legend_patch_5])
#plt.show()
plt.savefig("boxplot.tiff")

print("done")
