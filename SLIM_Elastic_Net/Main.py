from Cython import SLI
import numpy as np


icm = np.load("/home/luca/PycharmProjects/rec-sys-project/files/icm.npz")
urm = np.load("/home/luca/PycharmProjects/rec-sys-project/files/urm.npz")

recommender = SLIM_Elastic_Net_Cython(icm, urm)
recommender.fit()
recommender.recommend(user= "1", n=5)
