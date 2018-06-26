import glob
import os
import shutil

root_path = '/media/leejeyeol/74B8D3C8B8D38750/Data/AI2018_FACE_faceswap'
files = glob.glob(os.path.join(root_path,'*.*'))
for i, file in enumerate(files):
    save_path = os.path.join(root_path, 'AI2018_FACE_faceswap', '%05d.%s'%(i, os.path.basename(file).split('.')[-1]))
    shutil.move(file, save_path)
