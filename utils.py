import numpy as np
import math
import random
import pandas as pd

import constant
from object.Movie import Movie
from object.Rating import Rating
from object.User import User


def get_user_from_id(users, id):
    if users.get(id) is not None: return users[id]
    return User(id, [])


def read_input():
    # read movie input
    movies_data = pd.read_csv("dataSource/movies.csv")
    movies = []
    for index in movies_data.index:
        movie_id = movies_data.movieId[index]
        name = movies_data.title[index]
        genres = movies_data.genres[index].split("|")
        movies.append(Movie(movie_id, name, genres))

    # read rating input
    ratings_data = pd.read_csv("dataSource/ratings.csv")
    users = {}
    for index in ratings_data.index:
        user_id = ratings_data.userId[index]
        movie_id = ratings_data.movieId[index]
        rating = ratings_data.rating[index]
        time_stamp = ratings_data.timestamp[index]
        rating = Rating(user_id, movie_id, rating, time_stamp)

        user = get_user_from_id(users, user_id)
        user.ratings[movie_id] = rating
        users[user_id] = user

    return [movies, users]


def nDCG_cal(user, movies_recommended):
    watched = [0 for i in range(len(movies_recommended))]
    for i in range(len(movies_recommended)):
        if user.ratings[movies_recommended[i].id] is not None:
            watched[i] = 1
    dng_list = [watched[i] / np.log2(2 + i) for i in range(len(movies_recommended))]
    dNG = np.sum(dng_list)

    sortedWatch = sorted(watched, reverse=True)
    i_dng_list = [sortedWatch[i] / np.log2(2 + i) for i in range(len(movies_recommended))]
    iDNG = np.sum(i_dng_list)

    return dNG / iDNG


def read_rating_input():
    r_cols = ['userId', 'movieId', 'rating', 'timestamp']

    ratings_base = pd.read_csv('dataSource/ratings.csv', sep=',', names=r_cols, encoding='latin-1')

    input_matrix = ratings_base.to_numpy()
    # print(input_matrix)

    users = {}
    for row in input_matrix:
        user_id = row[0]
        user = get_user_from_id(users, user_id)
        user.ratings.append(row)
        users[user_id] = user

    data_train = []
    data_test = []
    random.seed(0)
    for user_id in users:
        rating_count = len(users[user_id].ratings)
        user_rating = users[user_id].ratings.copy()
        random.shuffle(user_rating)
        for i in range(rating_count):
            if i < math.floor(rating_count * 0.8):
                data_train.append(user_rating[i])
            else:
                data_test.append(user_rating[i])

    # print(data_train)
    return data_train, data_test


def normalize(Y):
    output = Y.copy()
    users = output[:, 0]
    bias = np.zeros((constant.total_user_count,))
    for i in range(constant.total_user_count):
        ids = np.where(users == i)[0].astype(np.int32)
        ratings = output[ids, 2]
        mean = np.mean(ratings)
        if np.isnan(mean):
            mean = 0
        bias[i] = mean
        output[ids, 2] = ratings - bias[i]
    return output, bias
