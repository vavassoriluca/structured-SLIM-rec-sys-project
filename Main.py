from SLIM_Elastic_Net.Cython.SLIM_Elastic_Net_Cython import SLIM_Elastic_Net_Cython
from scipy import sparse as sps
from sklearn.model_selection import train_test_split


icm = sps.load_npz("/home/luca/PycharmProjects/rec-sys-project/files/icm.npz")
urm = sps.load_npz("/home/luca/PycharmProjects/rec-sys-project/files/urm.npz")

urm_train, urm_test = train_test_split(urm, test_size=0.2)

recommender = SLIM_Elastic_Net_Cython(icm, urm_train, epochs=50)
recommender.fit()

print(recommender.evaluateRecommendations(urm_test, at=5, minRatingsPerUser=1, exclude_seen=False))
