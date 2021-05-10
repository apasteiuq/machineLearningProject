import pandas as pd
from object.Movie import Movie
from object.Rating import Rating


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
    ratings = []
    for index in ratings_data.index:
        user_id = ratings_data.userId[index]
        movie_id = ratings_data.movieId[index]
        rating = ratings_data.rating[index]
        time_stamp = ratings_data.timestamp[index]
        ratings.append(Rating(user_id, movie_id, rating, time_stamp))

    return [movies, ratings]
