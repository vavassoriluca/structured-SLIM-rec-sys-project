'''
@author = Luca Vavassori
'''

import numpy as np
from scipy import  sparse as sps
import time

from algorythms.relevant_k import relevant_k_matrix


class ElasticNet:

    #def __init__(self, icm, urm, np.int64 epochs, np.float64 learn, np.float64 gamma, np.float64 beta):
    def __init__(self, icm):
        self.icm = icm
        '''
        self.urm = urm
        self.epochs = epochs
        self.learn = learn
        self.gamma = gamma
        self.beta = beta
        '''

    def fit(self):
        t0 = time.time()
        self.rel_k_mtrx = relevant_k_matrix(self.icm)
        print(self.rel_k_mtrx)
        print("Cython: ", time.time() - t0)

    def relevant_k(icm, i):
        col_i = icm[:, i]
        result = col_i.T.dot(icm)
        return result.indices


    def relevant_k_matrix(icm):

        rows, cols = icm.shape
        dataR, indicesR = [], []
        indptrR = np.zeros(cols + 1, dtype=np.int32)

        for i in range(cols):
            tempI = ElasticNet.relevant_k(icm, i)
            tempD = np.ones((tempI.shape[0]), dtype=np.int32)
            dataR = np.append(dataR, np.asarray(tempD))
            indicesR = np.append(indicesR, np.asarray(tempI))
            indptrR[i + 1] = indptrR[i] + tempD.shape[0]

        return sps.csc_matrix((dataR, indicesR, indptrR), shape=(rows, cols))

    def fitp(self):
        t0 = time.time()
        ElasticNet.relevant_k_matrix(self.icm)
        print("Python: ", time.time() - t0)


x = sps.rand(100000,10000,0.01, format='csc')
x.data = np.ones(len(x.data), dtype=np.int32)
e = ElasticNet(x)
print("START")
e.fitp()
e.fit()