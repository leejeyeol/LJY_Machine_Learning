import threading
import pygame
def calling():
    global counter
    threading.Timer(period, calling).start()
    print(counter)
    pygame.mixer.music.play()
    counter = counter + 1

music = r"D:\resources\time.mp3"
freq = 16000
bitsize = -16
channels = 1
buffer = 2048
pygame.mixer.init(freq,bitsize,channels,buffer)
pygame.mixer.music.load(music)

period = 1800
counter = 0
calling()

