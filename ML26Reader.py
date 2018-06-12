import numpy as np
import pandas as pd
from scipy import sparse as sps

'''
movies_df = pd.read_csv("data/ml-latest/movies.csv")
tags_df = pd.read_csv("data/ml-latest/tags.csv").dropna(axis=0)

tempKind, tempID = [], []

for i in range(len(movies_df.movieId)):
    temp = movies_df.genres[i].split("|")
    tempKind.extend(temp)
    tempID.extend([movies_df.movieId[i] for y in range(len(temp))])

movies_df = pd.DataFrame({"movieId": tempID, "genres": tempKind}).dropna(axis=0)

print('Starting create unique list')
movie_list = list(movies_df.movieId.unique())
kind_list = list(movies_df.genres.unique())
tag_list = list(tags_df.tag.unique())

print('Starting create rows and cols')
rows = list()
cols = list()

rows = movies_df.movieId.astype('category', categories=movie_list).cat.codes
cols = movies_df.genres.astype('category', categories=kind_list).cat.codes
data = np.ones(len(rows)).squeeze()
icm_genres = sps.csc_matrix((data, (rows, cols)), shape=(len(movie_list), len(kind_list)))

rows = tags_df.movieId.astype('category', categories=movie_list).cat.codes
cols = tags_df.tag.astype('category', categories=tag_list).cat.codes
data = np.ones(len(rows)).squeeze()
icm_tags = sps.csc_matrix((data, (rows, cols)), shape=(len(movie_list), len(tag_list)))


ratings_df = pd.read_csv("data/ml-latest/ratings.csv").dropna(axis=0)

print('Starting create unique list for urm')
user_list = list(ratings_df.userId.unique())
print(len(user_list))


cols = list()
rows = list()

cols = ratings_df.movieId.astype('category', categories=movie_list).cat.codes
rows = ratings_df.userId.astype('category', categories=user_list).cat.codes
data = ratings_df.rating.astype(float)

urm = sps.csc_matrix((data, (rows, cols)), shape=(len(user_list),len(movie_list)), dtype=np.float32)

print("SAVE")
sps.save_npz("files/urm26M.npz", urm)
sps.save_npz("files/icm26M_genres.npz", icm_genres)
sps.save_npz("files/icm26M_tags.npz", icm_tags)
print("FINE")
'''

