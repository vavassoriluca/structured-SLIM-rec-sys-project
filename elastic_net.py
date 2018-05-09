import numpy as np
from scipy import sparse as sps

from SLIM_Elastic_Net.SLIM_Elastic_Net import SLIM_Elastic_Net

'''
movies = [i.strip().split("::") for i in open('/home/luca/Scaricati/ml-10M100K/movies.dat', 'r').readlines()]

movies_df = pd.DataFrame(movies, columns = ['MovieID', 'Title', 'Kind'], dtype = int)

tempKind, tempID = [], []

for i in range(len(movies_df.MovieID)):
    temp = movies_df.Kind[i].split("|")
    tempKind.extend(temp)
    tempID.extend([movies_df.MovieID[i] for y in range(len(temp))])

movies_df = pd.DataFrame({"MovieID": tempID, "Kind": tempKind}).dropna(axis=0)

print('Starting create unique list')
movie_list = list(movies_df.MovieID.unique())
kind_list = list(movies_df.Kind.unique())
print(len(movie_list))
print(kind_list)


print('Starting create rows and cols')
rows = list()
cols = list()

cols = movies_df.MovieID.astype('category', categories=movie_list).cat.codes
rows = movies_df.Kind.astype('category', categories=kind_list).cat.codes
data = np.ones(len(rows)).squeeze()
icm = sps.csc_matrix((data, (rows, cols)), shape=(len(kind_list),len(movie_list)))

ratings = [i.strip().split("::") for i in open('/home/luca/Scaricati/ml-10M100K/ratings.dat', 'r').readlines()]

ratings_df = pd.DataFrame(ratings, columns = ['UserID', 'MovieID', 'Ratings', 'Timestamp'], dtype = int).dropna(axis=0)

print('Starting create unique list for urm')
user_list = list(ratings_df.UserID.unique())
print(len(user_list))


cols = list()
rows = list()

cols = ratings_df.MovieID.astype('category', categories=movie_list).cat.codes
rows = ratings_df.UserID.astype('category', categories=user_list).cat.codes
data = ratings_df.Ratings.astype(float)

urm = sps.csc_matrix((data, (rows, cols)), shape=(len(user_list),len(movie_list)), dtype=np.float32)
sps.save_npz("files/urm.npz", urm)
sps.save_npz("files/icm.npz", icm)
'''

movie_list = np.load('files/movies_list.npy')
user_list = np.load('files/user_list.npy')
kind_list = np.load('files/kind_list.npy')

train = sps.load_npz("files/train.npz")
test = sps.load_npz("files/test.npz")
icm = sps.load_npz("files/icm.npz")

el = SLIM_Elastic_Net(icm, train)
el.fit(epochs=50)
print(el.evaluateRecommendations(test))


'''
{'AUC': 0.14041332760923478, 'precision': 0.06498282770463865, 'recall': 0.0029702795654990254, 'map': 0.03951178210265244, 'NDCG': 0.008137593858484806, 'MRR': 0.14227723716847893}
'''
