import numpy as np
import matplotlib.pyplot as plt

colors = np.random.randint(255, size=(20, 3)) / 255
np.random.seed(2)


class Modified_Neighbour_SOM():
    def __init__(self, e, N, d=3, learningRate=0.8):
        self.shape = N
        self.dimension = d
        self.epoch = e
        self.learningRate = learningRate
        self.weights = np.random.rand(self.shape, self.shape, self.dimension)
        self.initialWeights = self.weights

    def Euclidean_distance(self, x, y):
        return np.linalg.norm(x - y, axis=2) ** 2

    def find_winning_neuron(self, e):
        index = np.argmin(e)
        index = np.unravel_index(index, (self.shape, self.shape))
        return index

    def neighborhood(self, k):
        initial = 10
        return initial * np.exp(-k / self.epoch)

    def Gaussian_function(self, d, k):
        return np.exp(-(d ** 2) / (2 * (self.neighborhood(k) ** 2)))

    def neighbours(self, winning_index, k):
        w = np.indices((self.shape, self.shape))
        w[0] = winning_index[0] - w[0]
        w[1] = winning_index[1] - w[1]
        dist = np.linalg.norm(w, axis=0)
        return self.Gaussian_function(dist, k)

    def train(self, color):
        for i in range(self.epoch):
            for c in range(len(color)):
                distance = self.Euclidean_distance(color[c], self.weights)
                winning_index = self.find_winning_neuron(distance)
                h = self.neighbours(winning_index, i + 1)
                h = np.dstack([h] * 3)
                self.weights = self.weights + (self.learningRate * h * (color[c] - self.weights))
        return self.weights

    def plot_colors(self):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
        fig.suptitle('Here Neighbouring Radius is Not Fixed and varies over time', fontsize=16)
        plt.sca(axs[0])
        plt.title("Initial")
        plt.imshow(self.initialWeights)
        plt.sca(axs[1])
        plt.title("Final")
        plt.imshow(self.weights)
        plt.savefig('modified-nr-som.png')
        plt.show()


N = 40
dimension = 3
lr = 0.8
epoch = 1000

neighbour_SOFM = Modified_Neighbour_SOM(epoch, N, dimension, lr)
neighbour_SOFM.train(colors)
neighbour_SOFM.plot_colors()
