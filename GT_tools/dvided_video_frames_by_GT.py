import os
import glob
import numpy as np
import LJY_utils

frame_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/full-JuneHong-Kim"
GT_path = "/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope/full-JuneHong-Kim_30320_31504_GT.npy"
save_anomaly_path = os.path.join(frame_path, "anomaly")
save_normal_path = os.path.join(frame_path, "normal")
LJY_utils.make_dir(save_anomaly_path)
LJY_utils.make_dir(save_normal_path)

frame_list = glob.glob(os.path.join(frame_path, "*.*"))
frame_list.sort()
GT = np.load(GT_path)[0]


for i in range(len(GT)):
    if GT[i] == 1:
        os.rename(frame_list[i], os.path.join(save_anomaly_path, os.path.basename(frame_list[i])))
    else:
        os.rename(frame_list[i], os.path.join(save_normal_path, os.path.basename(frame_list[i])))
print("done")