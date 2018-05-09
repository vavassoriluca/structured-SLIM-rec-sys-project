
"""
Created on 23/10/17
@author: Luca Vavassori, Alex Porciani
"""


from Base.Recommender import Recommender
from Base.Recommender_utils import check_matrix
from Base.Similarity_Matrix_Recommender import Similarity_Matrix_Recommender
from SLIM_Elastic_Net.Cython.SLIM_Elastic_Net_Cython import SLIM_Elastic_Net_Cython


class SLIM_Elastic_Net(Recommender, Similarity_Matrix_Recommender):
    """ ItemKNN recommender"""

    def __init__(self, ICM, URM_train):
        super(SLIM_Elastic_Net, self).__init__()

        # CSC is required during evaluation
        self.URM_train = check_matrix(URM_train, 'csc')
        self.ICM = check_matrix(ICM, 'csc')


    def fit(self, k=50, shrink=100, lamb=0.001, beta=0.001, gamma=0.0001, epochs= 50, normalize=True):

        self.elastic_net = SLIM_Elastic_Net_Cython(self.ICM, self.URM_train, l=lamb, b=beta, g=gamma, epochs=epochs)

        self.W_sparse = self.elastic_net.fit(topK=k, shrink=shrink, normalize=normalize, mode="cosine")
        self.W_sparse = check_matrix(self.W_sparse, 'csr')
