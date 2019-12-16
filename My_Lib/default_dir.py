import platform
import os
import LJY_utils
# default folder

def dd_call(name=''):
    if platform.system() == 'Windows':
        DataDir = r'D:data'
        SaveDir = r'D:save'
        ResultDir = r'D:result'
    elif platform.system() == 'Linux':
        DataDir = '/home/mlpa/data_4T/data'
        SaveDir = '/home/mlpa/data_4T/save'
        ResultDir = '/home/mlpa/data_4T/result'

    if name == '':
        return DataDir, SaveDir, ResultDir
    else :
        DataDir, SaveDir, ResultDir = os.path.join(DataDir, name), os.path.join(SaveDir, name), os.path.join(ResultDir, name)
        LJY_utils.make_dirs([DataDir, SaveDir, ResultDir])
        return DataDir, SaveDir, ResultDir

