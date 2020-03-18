class Robot:
    def __init__(self, map_):
        self.location = (0,0) # horizontal일때 좌측, vertical일때 위 기준
        self.state = 0 # 0: horizontal, 1: vertical
        self.map_ = map_
        self.queue = [] #(location, state, time)
        self.action = ["move", "rotate"]
        self.move = ["r","l","u","d"]
        self.rotate = [1,2,3,4]
        self.time = 0
    def search_around(self):
        for move in self.move:
            if self.satte




robot=Robot(1)
