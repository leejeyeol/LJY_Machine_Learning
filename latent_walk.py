


class Latent_Walk_Squre():
    def __init__(self, size, four_points ):
        self.maxrix_size = int(size)
        self.four_points = four_points
        self.matrix = self.make_matrix(self.maxrix_size, self.four_points)
        self.task_queue = []

    def make_matrix(self, size, four_points):
        latent_vectors = [[[] for _ in range(size)] for _ in range(size)]
        latent_vectors[0][0] = four_points[0]
        latent_vectors[0][size-1] = four_points[1]
        latent_vectors[size-1][0] = four_points[2]
        latent_vectors[size-1][size-1] = four_points[3]

