import pandas as pd
import numpy as np

ratings_list = [i.strip().split("::") for i in open('/home/luca/Scaricati/ml-10M100K/ratings.dat', 'r').readlines()]

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)

URM_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
print(URM_df)

R = URM_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

print(R_demeaned)