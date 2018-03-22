'''
@author = Luca Vavassori, Alex Porciani
'''

import numpy as np
cimport numpy as np
from scipy import sparse as sps
cimport cython
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

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef float vector_product(self, float[:] a, int[:] ind_a, float[:] b,  int[:] ind_b):

        cdef float result = 0.0
        cdef int i = 0
        cdef int j = 0
        cdef int len_a = len(a)
        cdef int len_b = len(b)

        while(i < len_a and j < len_b):
            if(ind_a[i] == ind_b[j]):
                result += a[i] * b[j]
            if(ind_a[i] < ind_b[j]):
                i += 1
            else:
                j += 1

        return result

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef float get_sparse_elem(self, float[:] data, int[:] indices, int n):

        cdef int i = 0
        cdef int len_data = len(data)

        while(i < len_data and indices[i] < n):
            if(indices[i] == n):
                return data[i]
            i += 1
        return 0.0



    cdef learning_process(self):

        print("\nStart learnign process")

        cdef int items = self.icm.shape[1]
        cdef int[:] users
        cdef int[:] Si_indices
        cdef float[:] Si_data
        cdef int[:] R_iu_indices
        cdef float[:] R_iu_data
        cdef float[:] R_iu
        cdef int i = 0
        cdef int j = 0
        cdef int u = 0
        cdef int ep = 0
        cdef float e = 0.0
        cdef float r = 0.0
        cdef float R_iuj = 0.0

        for ep in range(self.epochs):

            for i in range(items):

                print("\n\nITEM {}\n\n".format(i))

                # Copy URM and set column i to 0, take the original column X
                R_i = self.urm.copy()
                ri = R_i[:,i].copy()
                R_i.data[R_i.indptr[i]:R_i.indptr[i+1]] = 0
                R_i = R_i.tocsr()

                # Get the indices of cells of S to learn
                Si_indices = self.S.indices[self.S.indptr[i]:self.S.indptr[i + 1]]
                Si_data = self.S.data[self.S.indptr[i]:self.S.indptr[i + 1]]

                # Get the indices of the users who rated item i
                users = ri.copy().indices

                for u in users:
                    print("\nUSER {}\n".format(u))

                    r = 0.0
                    e = 0.0
                    R_iu_data = R_i.data[R_i.indptr[u]:R_i.indptr[u+1]]
                    R_iu_indices = R_i.indices[R_i.indptr[u]:R_i.indptr[u+1]]
                    r = Elastic_Net.vector_product(self, Si_data, Si_indices, R_iu_data, R_iu_indices)
                    e = ri[u,0] - r

                    for j in range(len(Si_indices)):
                        R_iuj = Elastic_Net.get_sparse_elem(self, R_iu_data, R_iu_indices, Si_indices[j])
                        Si_data[j] -= e*R_iuj*self.l + Si_data[j]*self.b + self.g
                        print("Item = {}, User = {}, error = {}, j = {}, new S[{},{}] = {} ".format(i,u,e,j,Si_indices[j],i,Si_data[j]))

                self.S.data[self.S.indptr[i]:self.S.indptr[i+1]:] = np.asarray(Si_data, dtype=np.float32)


    def fit(self, topK = 100, shrink=0, normalize = True, mode = "cosine"):

        sim_comp = Cosine_Similarity(self.icm, topK, shrink, normalize, mode)
        print("\nStart computing the relevant K items [ICM Similarity]\n")
        self.S = sim_comp.compute_similarity()
        print("\nRelevant K computed [ICM Similarity]")
        Elastic_Net.learning_process(self)


