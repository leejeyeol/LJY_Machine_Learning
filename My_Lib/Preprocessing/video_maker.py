import os
#image_folder = '/media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN'

#os.system('ffmpeg -r 1 -i /media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/MG/gan_batch_1_%06d.png -vcodec mpeg4 -vf scale="480:480" -framerate 1 -filter:v "setpts=PTS/8" -y /media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/gan_only.mp4')
#os.system('ffmpeg -r 1 -i /media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/MG/ours_batch_1_%06d.png -vcodec mpeg4 -vf scale="480:480" -framerate 1 -filter:v "setpts=PTS/8"   -y /media/leejeyeol/74B8D3C8B8D38750/Experiment/AEGAN/ours.mp4')

# os.system(r'ffmpeg -r 16  -i D:\experiments\HANON\figure_animation\train\fig_%05d.png -vcodec png -vf scale="480:480"  -filter:v "setpts=PTS" -y D:\experiments\HANON\figure_animation\train_x2.mov')
# os.system(r'ffmpeg -r 8  -i D:\experiments\HANON\figure_animation\train\fig_%05d.png -vcodec png -vf scale="480:480"  -filter:v "setpts=PTS" -y D:\experiments\HANON\figure_animation\train.mov')
# os.system(r'ffmpeg -r 8  -i D:\experiments\HANON\figure_animation\eval\fig_%05d.png -vcodec png -vf scale="480:480" -filter:v "setpts=PTS" -y D:\experiments\HANON\figure_animation\eval.mov')

os.system(r'ffmpeg -r 16  -i D:\experiments\HANON\figure_animation\train\fig_%05d.png  -vf scale="480:480"  -filter:v "setpts=PTS"  -vcodec libx264 -pix_fmt yuv420p -acodec libvo_aacenc -ab 128k -y D:\experiments\HANON\figure_animation\train_x2.mov')
os.system(r'ffmpeg -r 8  -i D:\experiments\HANON\figure_animation\train\fig_%05d.png  -vf scale="480:480"  -filter:v "setpts=PTS"  -vcodec libx264 -pix_fmt yuv420p -acodec libvo_aacenc -ab 128k -y D:\experiments\HANON\figure_animation\train.mov')
os.system(r'ffmpeg -r 8  -i D:\experiments\HANON\figure_animation\eval\fig_%05d.png  -vf scale="480:480"  -filter:v "setpts=PTS"  -vcodec libx264 -pix_fmt yuv420p -acodec libvo_aacenc -ab 128k -y D:\experiments\HANON\figure_animation\eval.mov')

