import os
import glob
import cv2
from PIL import Image
from PIL import ImageTk
import tkinter as tki
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename

import numpy as np

file_path = ""
file_name = ""
frame_path_list = []
GT_list = []
check_list = []

def progress_bar():
    global GT_list
    global check_list
    global i_frame
    img = np.zeros((9, len(GT_list), 3), np.uint8)
    for i in range(len(GT_list)):
        if check_list[i] == 0:
            img[3:6, i, 0] = 255
        else:
            if GT_list[i] == 0:
                img[6:9, i, 1] = 255
            else:
                img[0:3, i, 2] = 255
    #img[:, i_frame] = [0, 215, 255]
    return cv2.resize(img, (224, 9), cv2.INTER_AREA)

def save_GT():
    global file_path
    global GT_list
    global check_list
    global i_frame
    save_path = os.path.join(os.path.dirname(file_path),(os.path.basename(file_path) + "_%d_%d_GT.npy" % (i_frame, len(GT_list))))
    GT = []
    GT.append(GT_list)
    GT.append(check_list)

    np.save(save_path, np.asarray(GT))
    print("save gt")

def load_GT():
    global file_path
    global frame_path_list
    global GT_list
    global check_list
    global i_frame
    file_path = askopenfilename(initialdir=os.path.dirname(file_path))
    if os.path.basename(file_path).split('_')[-1].split('.')[0] == "GT":
        if int(os.path.basename(file_path).split('_')[-2]) == len(frame_path_list):
            i_frame = int(os.path.basename(file_path).split('_')[-3])
            GT = np.load(file_path)
            GT_list = GT[0]
            check_list = GT[1]
            show_image()
        else:
            print("error... lenght is not matched")
    else:
        print("error... this file is not GT")

def select_frames_path():
    global file_name
    global file_path
    global frame_path_list
    global GT_list
    global check_list
    file_path = askdirectory(initialdir="/media/leejeyeol/74B8D3C8B8D38750/Data/endoscope")
    frame_path_list = glob.glob(os.path.join(file_path, "*.*"))
    frame_path_list.sort()
    file_name = os.path.basename(file_path)
    GT_list = [0] * len(frame_path_list)
    check_list = [0] * len(frame_path_list)
    show_image()

def show_next_image(event):
    global i_frame
    global check_list
    global frame_path_list
    if i_frame + 1 > len(frame_path_list) - 1:
        print("error... frame number out of range")
    else:
        check_list[i_frame] = 1
        i_frame = i_frame + 1
        show_image()

def show_past_image(event):
    global i_frame
    global check_list
    global frame_path_list
    if i_frame + 1 < 0:
        print("error... frame number out of range")
    else:
        check_list[i_frame] = 1
        i_frame = i_frame - 1
        show_image()

def check_GT_as_normal(event):
    global i_frame
    global GT_list
    global check_list
    GT_list[i_frame] = 0
    check_list[i_frame] = 1
    show_image()

def check_GT_as_anomaly(event):
    global i_frame
    global GT_list
    global check_list
    GT_list[i_frame] = 1
    check_list[i_frame] = 1
    show_image()

def go_to_frame(event):
    global i_frame
    global frame_path_list
    frame = int(e1.get())
    if frame >=0 and frame <= len(frame_path_list)-1:
        i_frame = frame
        e1.delete(0, 'end')
        show_image()
    else:
        e1.delete(0, 'end')
        print("error... frame number out of range")


def show_image():
    global file_name
    global panel_cur, panel_past, panel_next, panel_progress, cur_frame_label
    global frame_path_list
    global GT_list
    global i_frame
    cur_frame = cv2.imread(frame_path_list[i_frame])

    if i_frame-1 < 0:
        past_frame = np.zeros((cur_frame.shape[0], cur_frame.shape[1], cur_frame.shape[2]), np.uint8)
    else:
        past_frame = cv2.imread(frame_path_list[i_frame-1])
    if i_frame+1 > len(frame_path_list)-1:
        next_frame = np.zeros((cur_frame.shape[0], cur_frame.shape[1], cur_frame.shape[2]), np.uint8)
    else:
        next_frame = cv2.imread(frame_path_list[i_frame+1])

    if GT_list[i_frame] == 1:
        color = (0, 0, 200)
    else:
        color = (0, 200, 0)
    cv2.rectangle(cur_frame,(0, 0),(cur_frame.shape[0]-1, cur_frame.shape[1]-1), color, 4)

    prog_bar = progress_bar()

    cur_frame = Image.fromarray(cur_frame[...,::-1])
    cur_frame = ImageTk.PhotoImage(cur_frame)
    past_frame = Image.fromarray(past_frame[...,::-1])
    past_frame = ImageTk.PhotoImage(past_frame)
    next_frame = Image.fromarray(next_frame[...,::-1])
    next_frame = ImageTk.PhotoImage(next_frame)
    prog_bar = Image.fromarray(prog_bar[...,::-1])
    prog_bar = ImageTk.PhotoImage(prog_bar)

    if panel_cur is None or panel_next is None or panel_past is None or cur_frame_label is None or panel_progress is None:
        panel_cur = tki.Label(image=cur_frame)
        panel_cur.image = cur_frame
        panel_cur.grid(row=3, column=1, columnspan=2)

        panel_past = tki.Label(image=past_frame)
        panel_past.image = past_frame
        panel_past.grid(row=3, column=0)

        panel_next = tki.Label(image=next_frame)
        panel_next.image = next_frame
        panel_next.grid(row=3, column=3)

        panel_progress = tki.Label(image=prog_bar)
        panel_progress.image = prog_bar
        panel_progress.grid(row=1, column=1, columnspan=2)


        cur_frame_label=tki.Label(root, text=file_name+"\n\nCurrent frame : %d / %d" % (i_frame+1, len(frame_path_list))).grid(row=0, column=0)
    else:
        panel_cur.configure(image=cur_frame)
        panel_cur.image = cur_frame
        panel_past.configure(image=past_frame)
        panel_past.image = past_frame
        panel_next.configure(image=next_frame)
        panel_next.image = next_frame
        panel_progress.configure(image=prog_bar)
        panel_progress.image = prog_bar

        cur_frame_label.configure(text=file_name+"\n\nCurrent frame : %d / %d" % (i_frame+1, len(frame_path_list)))



root = tki.Tk()
cur_frame_label = None
panel_past = None
panel_cur = None
panel_next = None
panel_progress = None
i_frame = 0

select_frames_path()


e1 = tki.Entry(root)
e1.grid(row=0, column=1)
select_file_btn = tki.Button(root, text="Find new path", command=select_frames_path)
select_file_btn.grid(row=0, column=3)
select_file_btn = tki.Button(root, text="Go to frame", command=lambda: go_to_frame(1))
select_file_btn.grid(row=0, column=2)

save_btn = tki.Button(root, text="Save GT", command=save_GT)
save_btn.grid(row=1, column=0)
save_btn = tki.Button(root, text="Load GT", command=load_GT)
save_btn.grid(row=1, column=3)

show_btn = tki.Label(root, text="Next\n-->")
show_btn.grid(row=5, column=3, rowspan=2)
show_btn = tki.Label(root, text="Past\n<--")
show_btn.grid(row=5, column=0, rowspan=2)
show_btn = tki.Label(root, text="^\nset anomaly")
show_btn.grid(row=5, column=1, columnspan=2)
show_btn = tki.Label(root, text="set normal\nv")
show_btn.grid(row=6, column=1, columnspan=2)

root.bind('<Left>', show_past_image)
root.bind('<Right>', show_next_image)
root.bind('<Up>', check_GT_as_anomaly)
root.bind('<Down>', check_GT_as_normal)
root.bind('<Return>', go_to_frame)

root.title("Ground Truth Maker - anomaly detection with video")

root.mainloop()



