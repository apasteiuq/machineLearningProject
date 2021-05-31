import numpy as np

import constant
import utils


class KNN:
    def __init__(self, Y, k):
        self.Y = Y
        self.k = k
        self.user_count = constant.total_user_count
        self.movie_count = constant.total_movie_count
        self.rating_count = Y.shape[0]
        self.matrix_Y = np.zeros((self.user_count, self.movie_count))
        self.create_matrix_y()

        self.neighbors = np.zeros((self.user_count, k))

    def create_matrix_y(self):
        for i in range(self.rating_count):
            self.matrix_Y[self.Y[i, 0], self.Y[i, 1]] = self.Y[i, 2]

    def get_k_nearest_neighbor(self, u):
        u = int(u)
        if np.sum(self.neighbors[u]) != 0:
            return [self.matrix_Y[int(self.neighbors[u, i]), :] for i in range(self.k)]
        dist = np.zeros(self.user_count)
        for i in range(self.user_count):
            dist[i] = np.linalg.norm(self.matrix_Y[u, :] - self.matrix_Y[i, :])
        neighbors_id = np.argsort(dist)[1:self.k+1]
        self.neighbors[u] = neighbors_id
        neighbors = [self.matrix_Y[i, :] for i in neighbors_id]
        return neighbors

    def predict(self, u, m):
        neighbors = self.get_k_nearest_neighbor(u)
        m_rating = [neighbor[m] for neighbor in neighbors]
        return np.mean(m_rating)

    def cal_RMSE(self, d_test):
        test_count = d_test.shape[0]
        SE = 0
        for n in range(test_count):
            prediction = self.predict(d_test[n, 0], d_test[n, 1])
            SE += (prediction - d_test[n, 2]) ** 2

        RMSE = np.sqrt(SE/test_count)
        return RMSE

