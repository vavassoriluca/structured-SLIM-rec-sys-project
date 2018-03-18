'''
@author = Luca Vavassori
'''

import numpy as np
from scipy import  sparse as sps
import time


class ElasticNet:

    def __init__(self, icm, urm, epochs, learn, gamma, beta):

        self.icm = icm
        self.urm = urm
        self.epochs = epochs
        self.learn = learn
        self.gamma = gamma
        self.beta = beta


    # --------------------------
    #      Relevant K Items
    # --------------------------

    def relevant_k(icm, i):

        col_i = icm[:, i]
        result = col_i.T.dot(icm)
        return result.data, result.indices

    def relevant_k_matrix(icm):

        rows, cols = icm.shape
        dataR, indicesR = [], []
        indptrR = np.zeros(cols + 1, dtype=np.int32)

        for i in range(cols):
            tempD, tempI = ElasticNet.relevant_k(icm, i)
            dataR = np.append(dataR, np.asarray(tempD))
            indicesR = np.append(indicesR, np.asarray(tempI))
            indptrR[i + 1] = indptrR[i] + tempD.shape[0]

        return sps.csc_matrix((dataR, indicesR, indptrR), shape=(cols, cols))


    # --------------------------
    #   Similarity Computation
    # --------------------------

    def similarity_init(self):

        self.similarity_mtrx = self.rel_k_mtrx.copy()
        nnz = self.similarity_mtrx.nnz
        self.similarity_mtrx.data = np.random.uniform(0.0, 1.0, nnz)

    def epoch(self):
        # TODO

    def similarity_learning(self):

        ElasticNet.similarity_init(self)
        for _ in range(self.epochs):
            ElasticNet.epoch(self)


    # --------------------------
    #       Public Methods
    # --------------------------

    def fit(self):

        t0 = time.time()
        self.rel_k_mtrx = ElasticNet.relevant_k_matrix(self.icm)
        print("Creation of relevant K items matrix: ", time.time() - t0)
