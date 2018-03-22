'''
@author = Luca Vavassori, Alex Porciani
'''

import numpy as np
cimport numpy as np
from scipy import sparse as sps
import datetime

from Base.Cython.cosine_similarity import Cosine_Similarity

cdef class Elastic_Net:

    cdef S
    cdef urm
    cdef icm
    cdef float l
    cdef float b
    cdef float g
    cdef long epochs

    def __init__(self, icm, urm, double l=0.001, double b=0.001, double g=0.001, int epochs=10):

        self.icm = icm
        self.urm = urm
        self.l = l
        self.b = b
        self.g = g
        self.epochs = epochs


    cdef learning_process(self):

        print("\nStart learnign process")

        cdef int items = self.icm.shape[1]
        cdef int[:] users
        cdef int[:] Si_indices
        cdef int i = 0
        cdef int j = 0
        cdef int u = 0
        cdef int ep = 0
        cdef float e = 0.0
        cdef float r = 0.0

        for ep in range(self.epochs):

            for i in range(items):

                print("\n\nITEM {}\n\n".format(i))

                # Copy URM and set column i to 0, take the original column X
                R_i = self.urm.copy()
                ri = R_i[:,i].copy()
                R_i.data[R_i.indptr[i]:R_i.indptr[i+1]] = 0
                R_i.tocsr()

                # Get the indices of cells of S to learn
                Si_indices = self.S.indices[self.S.indptr[i]:self.S.indptr[i + 1]]

                # Get the indices of the users who rated item i
                users = ri.copy().indices

                for u in users:
                    r = 0.0
                    e = 0.0
                    r = R_i[u,:].dot(self.S[:, i])[0,0]
                    e = ri[u,0] - r
                    for j in Si_indices:
                        self.S[j,i] -= e*R_i[u,j]*self.l + self.S[j,i]*self.b + self.g
                        print("Item = {}, User = {}, error = {}, j = {}, new S[j,i] = {} ".format(i,u,e,j,self.S[j,i]))













    def fit(self, topK = 100, shrink=0, normalize = True, mode = "cosine"):

        sim_comp = Cosine_Similarity(self.icm, topK, shrink, normalize, mode)
        print("\nStart computing the relevant K items [ICM Similarity]\n")
        self.S = sim_comp.compute_similarity()
        print("\nRelevant K computed [ICM Similarity]")
        Elastic_Net.learning_process(self)


