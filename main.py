import utils
from model.MatrixFactorial import MatrixFactorial
from model.KNN import KNN
import numpy as np

data_train, data_test = utils.read_rating_input()

d_train = np.array(data_train, dtype=int)
d_test = np.array(data_test, dtype=int)

d_train[:, :2] -= 1
d_test[:, :2] -= 1


def use_MF():
    mf = MatrixFactorial(d_train, 3, lambd=.1, print_every=10, learning_rate=3, max_iter=50)
    mf.fit()
    RMSE = mf.cal_RMSE(d_test)
    print('\nMF, RMSE =', RMSE)


knn = KNN(d_train, 5)
RMSE = knn.cal_RMSE(d_test)
print('\nKNN, RMSE =', RMSE)
