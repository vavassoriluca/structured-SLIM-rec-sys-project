'''
@author = Luca Vavassori, Alex Porciani
'''

import numpy as np
cimport numpy as np
from scipy import sparse as sps
cimport cython
from Base.Cython.cosine_similarity import Cosine_Similarity
from Base.metrics import roc_auc, precision, recall, rr, map, ndcg
import time

cdef class SLIM_Elastic_Net_Cython:

    cdef S
    cdef int[:] S_indices
    cdef int[:] S_indptr
    cdef float[:] S_data

    cdef int[:] R_i_indices
    cdef int[:] R_i_indptr
    cdef float[:] R_i_data

    cdef int[:] test_indices
    cdef int[:] test_indptr
    cdef float[:] test_data

    cdef urm_train
    cdef urm_test
    cdef icm
    cdef float l
    cdef float b
    cdef float g
    cdef long epochs
    cdef int at
    cdef int min_ratings_per_user
    cdef int exclude_seen


    def __init__(self, icm, urm, double l=0.001, double b=0.001, double g=0.001, int epochs=50):

        self.icm = icm
        self.urm_train = urm
        self.l = l
        self.b = b
        self.g = g
        self.epochs = epochs


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
        cdef int sample_index, index_inner, item_id
        cdef int current_item = 0
        cdef int j = 0
        cdef int k = 0
        cdef int z = 0
        cdef int user_id = 0
        cdef int current_epoch
        cdef float error = 0.0
        cdef float prediction = 0.0
        cdef float Si_data_prev = 0.0
        cdef long time_item = 0
        cdef long time_epoch = 0

        R_i_csc = self.urm_train.copy()
        R_i = self.urm_train.tocsr()

        self.R_i_indices = R_i.indices
        self.R_i_indptr = R_i.indptr
        self.R_i_data = R_i.data

        for current_item in range(items):

            print("\n\nITEM {}\n\n".format(current_item))
            time_item = time.time()

            # Copy URM and set column i to 0, take the original column X
            ri = R_i_csc[:, current_item].copy()
            #R_i.data[R_i.indptr[current_item]:R_i.indptr[current_item + 1]] = 0

            # Get the indices of cells of S to learn
            Si_indices = self.S_indices[self.S_indptr[current_item]:self.S_indptr[current_item + 1]]
            Si_data = self.S_data[self.S_indptr[current_item]:self.S_indptr[current_item + 1]]

            # Get the indices of the users who rated item i

            ri_data = ri.data
            ri_indices = np.array(ri.indices, dtype=np.int32)


            for index_inner in range(len(Si_indices)):

                item_id = Si_indices[index_inner]

                Si_mask[item_id] = True
                Si_dense[item_id] = Si_data[index_inner]


            for current_epoch in range(self.epochs):

                #print("\n\nEPOCH {}".format(current_epoch))

                #time_epoch = time.time()

                #print("to begin {}: {}".format(current_epoch, time.time() - time_epoch))

                for sample_index in range(len(ri_indices)):

                    user_id = ri_indices[sample_index]

                    R_iu_data = self.R_i_data[self.R_i_indptr[user_id]:self.R_i_indptr[user_id+1]]
                    R_iu_indices = self.R_i_indices[self.R_i_indptr[user_id]:self.R_i_indptr[user_id+1]]

                    #print("init {}: {}".format(current_epoch, time.time() - time_epoch))

                    #prediction = Elastic_Net.vector_product(self, R_iu_data, R_iu_indices, Si_dense, Si_mask)

                    prediction = 0.0

                    for index_inner in range(len(R_iu_indices)):

                        item_id = R_iu_indices[index_inner]

                        if Si_mask[item_id] == True and item_id != current_item:
                            prediction += R_iu_data[index_inner] * Si_dense[item_id]


                    error = prediction - ri_data[sample_index]

                    #print("dot {}: {}".format(current_epoch, time.time() - time_epoch))


                    for index_inner in range(len(R_iu_indices)):

                        item_id = R_iu_indices[index_inner]

                        if Si_mask[item_id] == True:
                            Si_data_prev = Si_dense[item_id]
                            Si_dense[item_id] -= error * R_iu_data[index_inner] * self.l + Si_dense[item_id] * self.b + self.g
                            #print("Si_prev: {}, Si_new: {}, e: {}, R_iu: {}".format(Si_data_prev, Si_dense[j], e, R_iu_data[z]))

                    #print("update {}: {}".format(current_epoch, time.time() - time_epoch))

                    #input()


                for index_inner in range(len(Si_indices)):
                    Si_data[index_inner] = Si_dense[Si_indices[index_inner]]



            self.S_data[self.S_indptr[current_item]:self.S_indptr[current_item + 1]:] = Si_data

            print("Elapsed time item {}: {} s, sample/sec {:.2f}".format(current_item, time.time() - time_item, len(ri_indices)*self.epochs/(time.time() - time_item)))



            for index in range(len(Si_indices)):

                item_id = Si_indices[index]

                Si_mask[item_id] = False
                Si_dense[item_id] = 0.0




            print("Elapsed time item {}: {}".format(current_item, time.time() - time_item))

        return sps.csc_matrix((self.S_data, self.S_indices, self.S_indptr), shape=(items,items))


    def fit(self, topK = 100, shrink=100, normalize = True, mode = "cosine"):

        sim_comp = Cosine_Similarity(self.icm, topK, shrink, normalize, mode)
        print("\nStart computing the relevant K items [ICM Similarity]\n")
        self.S = sim_comp.compute_similarity()
        self.S_indices = self.S.indices
        self.S_data = self.S.data
        self.S_indptr = self.S.indptr
        print("\nRelevant K computed [ICM Similarity]")
        return SLIM_Elastic_Net_Cython.learning_process(self)
