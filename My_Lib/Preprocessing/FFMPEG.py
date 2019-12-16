import os
import subprocess as sp

FFMPEG_BIN = "ffmpeg"
target_rows = 255
target_cols = 255

def make_dir(path):
    # if there is no directory, make a directory.
    if not os.path.exists(path):
        os.makedirs(path)
        print(path)
    return

# video --> frames =====================================================================================================
print('Extract images...')
#back young ju (01104575) 19 Jun 13_1
filename = "Lee_Sungmu.mpg"
folder_path ="/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope"

# output directory
output_dir = os.path.join(folder_path, os.path.splitext(filename)[0])
make_dir(output_dir)


# run ffmpeg command
print('\tFrom: ' + filename)
command = [FFMPEG_BIN,
           '-i', os.path.join(folder_path, filename),
           '-s', str(target_rows) + 'x' + str(target_cols),  # [rows x cols]
           '-pix_fmt', 'rgb24',
            '-vf', 'fps=30',
           os.path.join(output_dir, 'frame_%07d.png')]
sp.call(command)  # call command
print("Extraction is done")
