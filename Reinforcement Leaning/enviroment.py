import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage


UNIT = 50

road = 0
wall = -100
goal = 1
initial_state = [0, 6]
reward_map = np.array([[road,road,goal,wall,road,road,road],
                [road,wall,road,wall,road,wall,wall],
                [road,wall,road,wall,road,road,road],
                [road,wall,road,wall,road,wall,road],
                [road,wall,road,road,road,wall,road],
                [road,wall,road,wall,road,wall,road],
                [road,road,road,wall,road,road,road]])
HEIGHT = reward_map.shape[0]
WIDTH = reward_map.shape[1]



class enviroment(tk.Tk):
    def __init__(self):
        super(enviroment, self).__init__()
        self.inital_state = [0, 6]
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.title("Reinforcement - Maze")
        self.reward_map = reward_map
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shape = self.load_image()
        self.canvas = self._build_canvas()
        self._build_map()
        self.goal = []
        self.player = []

        self.step_counter = 0
        self.update()
        print(1)

    def load_image(self):
        rectangle = PhotoImage(
            Image.open("./rectangle.png").resize((30, 30)))
        triangle = PhotoImage(
            Image.open("./triangle.png").resize((30, 30)))
        circle = PhotoImage(
            Image.open("./circle.png").resize((30, 30)))
        return rectangle, triangle, circle

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white', height=HEIGHT*UNIT, width=WIDTH*UNIT)
        for column in range(0, WIDTH*UNIT, UNIT):
            x0, y0, x1, y1 = column, 0, column, HEIGHT*UNIT
            canvas.create_line(x0, y0, x1, y1)
        for row in range(0, HEIGHT*UNIT, UNIT):
            x0, y0, x1, y1 = 0, row, HEIGHT*UNIT, row
            canvas.create_line(x0, y0, x1, y1)
        canvas.pack()

        return canvas

    def _build_map(self):
        self.player = self.canvas.create_image((UNIT * self.inital_state[1]) + UNIT / 2,
                                 (UNIT * self.inital_state[0]) + UNIT / 2,
                                 image=self.shape[1])

        for x in range(HEIGHT):
            for y in range(WIDTH):
                if reward_map[y][x] == road:
                    pass
                elif reward_map[y][x] == wall:
                    self.canvas.create_image((UNIT*x) + UNIT/2,
                                             (UNIT*y) + UNIT/2,
                                             image=self.shape[0])
                elif reward_map[y][x] == goal:
                    self.canvas.create_image((UNIT*x) + UNIT/2,
                                             (UNIT*y) + UNIT/2,
                                             image=self.shape[2])
env = enviroment()