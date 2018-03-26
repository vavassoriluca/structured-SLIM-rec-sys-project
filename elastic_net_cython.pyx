'''
@author = Luca Vavassori, Alex Porciani
'''

import numpy as np
cimport numpy as np
from scipy import sparse as sps
cimport cython
from Base.Cython.cosine_similarity import Cosine_Similarity
import time

cdef class Elastic_Net:

    cdef S
    cdef int[:] S_indices
    cdef int[:] S_indptr
    cdef float[:] S_data
    cdef int[:] R_i_indices
    cdef int[:] R_i_indptr
    cdef float[:] R_i_data
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
    cdef float vector_product(self, float[:] a, int[:] ind_a, float[:] b,  int[:] mask_b):

        cdef float result = 0.0
        cdef int i = 0
        cdef int j = 0

        for i in ind_a:

            if mask_b[i] == 1:
                result += a[j] * b[i]

            j += 1

        return result


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef learning_process(self):

        print("\nStart learnign process")

        cdef int items = self.icm.shape[1]
        cdef int[:] Si_mask = np.zeros(items, np.int32)
        cdef float[:] Si_dense = np.zeros(items, np.float32)
        cdef int[:] Si_indices
        cdef float[:] Si_data
        cdef int[:] R_iu_indices
        cdef float[:] R_iu_data
        cdef int[:] ri_indices
        cdef float[:] ri_data
        cdef float[:] R_iu
        cdef int index, index_inner
        cdef int i = 0
        cdef int j = 0
        cdef int k = 0
        cdef int z = 0
        cdef int s = 0
        cdef int ep = 0
        cdef float e = 0.0
        cdef float r = 0.0
        cdef float Si_data_prev = 0.0
        cdef long time_item = 0
        cdef long time_epoch = 0

        for i in range(items):

            print("\n\nITEM {}\n\n".format(i))
            time_item = time.time()

            # Copy URM and set column i to 0, take the original column X
            R_i = self.urm.copy()
            ri = R_i[:,i].copy()
            R_i.data[R_i.indptr[i]:R_i.indptr[i+1]] = 0
            R_i = R_i.tocsr()
            self.R_i_indices = R_i.indices
            self.R_i_indptr = R_i.indptr
            self.R_i_data = R_i.data


            # Get the indices of cells of S to learn
            Si_indices = self.S_indices[self.S_indptr[i]:self.S_indptr[i + 1]]
            Si_data = self.S_data[self.S_indptr[i]:self.S_indptr[i + 1]]

            # Get the indices of the users who rated item i

            ri_data = ri.data
            ri_indices = np.array(ri.indices, dtype=np.int32)

            z = 0

            for k in Si_indices:

                Si_mask[k] = 1
                Si_dense[k] = Si_data[z]
                z += 1

            for ep in range(self.epochs):

                print("\n\nEPOCH {}".format(ep))
                time_epoch = time.time()

                for index in range(len(ri_indices)):

                    #print("hello {}: {}".format(ep, time.time() - time_epoch))

                    s = ri_indices[index]

                    #print("s {}: {}".format(ep, time.time() - time_epoch))

                    r = 0.0
                    e = 0.0
                    R_iu_data = self.R_i_data[self.R_i_indptr[s]:self.R_i_indptr[s+1]]
                    #print("init {}: {}".format(ep, time.time() - time_epoch))

                    R_iu_indices = self.R_i_indices[self.R_i_indptr[s]:self.R_i_indptr[s+1]]

                    #print("init {}: {}".format(ep, time.time() - time_epoch))

                    r = Elastic_Net.vector_product(self, R_iu_data, R_iu_indices, Si_dense, Si_mask)
                    e = r - ri_data[s]

                    #print("dot {}: {}".format(ep, time.time() - time_epoch))

                    z = 0
                    for index_inner in range(len(R_iu_indices)):

                        j = R_iu_indices[index_inner]

                        if Si_mask[j] == 1:
                            Si_data_prev = Si_dense[j]
                            Si_dense[j] -= e*R_iu_data[z]*self.l + Si_dense[j]*self.b + self.g
                            #print("Si_prev: {}, Si_new: {}, e: {}, R_iu: {}".format(Si_data_prev, Si_dense[j], e, R_iu_data[z]))
                        z += 1

                    z = 0
                    for z in range(len(Si_indices)):
                        Si_data[z] = Si_dense[Si_indices[z]]

                    #print("update {}: {}".format(ep, time.time() - time_epoch))

                    #input()


                print("Elapsed time epoch {}: {}".format(ep, time.time() - time_epoch))

            self.S_data[self.S_indptr[i]:self.S_indptr[i+1]:] = Si_data

            for k in Si_indices:

                Si_mask[k] = 0
                Si_dense[k] = 0

            print("Elapsed time item {}: {}".format(i, time.time() - time_item))


    def fit(self, topK = 100, shrink=0, normalize = True, mode = "cosine"):

        sim_comp = Cosine_Similarity(self.icm, topK, shrink, normalize, mode)
        print("\nStart computing the relevant K items [ICM Similarity]\n")
        self.S = sim_comp.compute_similarity()
        self.S_indices = self.S.indices
        self.S_data = self.S.data
        self.S_indptr = self.S.indptr
        print("\nRelevant K computed [ICM Similarity]")
        Elastic_Net.learning_process(self)


