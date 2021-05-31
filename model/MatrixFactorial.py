import numpy as np
import constant
import utils


class MatrixFactorial:
    def __init__(self, Y, k, lambd=0.1, learning_rate=0.5, max_iter=1000, print_every=100):
        self.raw_Y = Y
        self.k = k
        self.lambd = lambd
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every

        self.user_count = constant.total_user_count
        self.movie_count = constant.total_movie_count
        self.rating_count = Y.shape[0]

        self.X = np.random.randn(self.movie_count, k)
        self.W = np.random.randn(k, self.user_count)
        self.Y_normalized, self.user_bias = utils.normalize(Y)

    def cal_residual_square(self, rating_id):
        user_id = int(self.Y_normalized[rating_id, 0])
        movie_id = int(self.Y_normalized[rating_id, 1])
        rating = self.Y_normalized[rating_id, 2]
        return 0.5 * (rating - self.W[:, user_id].dot(self.X[movie_id, :])) ** 2

    # loss_cal using Frobenius norm
    def loss(self):
        ans = np.sum([self.cal_residual_square(i) for i in range(self.rating_count)])
        ans /= self.rating_count
        ans += 0.5 * self.lambd * (np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))
        return ans

    def get_items_rated_by_user(self, user_id):
        ids = np.where(self.Y_normalized[:, 0] == user_id)[0]
        item_ids = self.Y_normalized[ids, 1].astype(np.int32)
        ratings = self.Y_normalized[ids, 2]
        return item_ids, ratings

    def get_users_who_rate_item(self, item_id):
        ids = np.where(self.Y_normalized[:, 1] == item_id)[0]
        user_ids = self.Y_normalized[ids, 0].astype(np.int32)
        ratings = self.Y_normalized[ids, 2]
        return user_ids, ratings

    def updateX(self):
        for m in range(self.movie_count):
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T) / self.rating_count + self.lambd * self.X[m, :]
            self.X[m, :] -= self.learning_rate * grad_xm.reshape((self.k,))

    def updateW(self):
        for n in range(self.user_count):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            # gradient
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n])) / self.rating_count + self.lambd * self.W[:, n]
            self.W[:, n] -= self.learning_rate * grad_wn.reshape((self.k,))

    def predict(self, u, m):
        u = int(u)
        m = int(m)
        bias = self.user_bias[u]
        prediction = self.X[m, :].dot(self.W[:, u]) + bias
        return min(max(prediction, 0), 5)

    def cal_RMSE(self, d_test):
        test_count = d_test.shape[0]
        SE = 0
        for n in range(test_count):
            prediction = self.predict(d_test[n, 0], d_test[n, 1])
            SE += (prediction - d_test[n, 2]) ** 2

        RMSE = np.sqrt(SE/test_count)
        return RMSE

    def cal_RSS(self, d_test):
        test_rating_count = d_test.shape[0]
        RSS = 0
        for i in range(test_rating_count):
            prediction = self.predict(d_test[i, 0], d_test[i, 1])
            RSS += (prediction - d_test[i, 2]) ** 2

        return RSS

    def fit(self):
        for i in range(self.max_iter):
            self.updateX()
            self.updateW()
            if (i + 1) % self.print_every == 0:
                rmse_train = self.cal_RMSE(self.raw_Y)
                print('iter =', i + 1, ', loss =', self.loss(), ', RMSE train =', rmse_train)
