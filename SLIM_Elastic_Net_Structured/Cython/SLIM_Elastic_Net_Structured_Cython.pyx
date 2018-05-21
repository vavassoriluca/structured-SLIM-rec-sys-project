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
import sys

cdef class SLIM_Elastic_Net_Structured_Cython:

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

        '''
        :param icm: ICM matrix used to generate the similarity matrix structure
        :param urm: URM matrix used to train the model
        :param l: learning rate
        :param b: beta coefficient of L2 norm regularization factor
        :param g: gamma coefficient of L1 norm regularization factor
        :param epochs: number of iteration of the learning process
        '''

        self.icm = icm
        self.urm_train = urm
        self.l = l
        self.b = b
        self.g = g
        self.epochs = epochs


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    cdef learning_process(self):

        '''
        For each item we apply the learning process as many times as the number of epochs.
        The learning process consists of 
                
        :param items: number of items of the icm matrix
        :param Si_mask: mask used to filter Si_dense based on the indices of the sparse matrix
        :param Si_dense: dense version of the i-th row of the similarity matrix
        :param Si_indices: indices of the i-th row of the sparse similarity matrix
        :param Si_data: data of the i-th row of the sparse similarity matrix
        :param R_iu_indices: indices of the URM sparse matrix related to the u-th row (u-th user)
        :param R_iu_data: data of the URM sparse matrix related to the u-th row (u-th user)
        :param ri_indices: indices of the URM sparse matrix related to the i-th column (i-th item)
        :param ri_data: data of the URM sparse matrix related to the i-th column (i-th item)
        :param sample_index: index of the loop in which R_iu is multiplied by Si
        :param index_inner: index used in the loop to fill Si_dense and Si_mask
        :param item_id: id of the item used to fill Si_dense and Si_mask
        :param current_item: id of the current item in the main loop
        :param user_id: id of the user in the sample index loop
        :param current_epoch: epoch of the loop
        :param error: error computed by subtract the 
        :param prediction: the predicted rating
        :param time_item: starting time for the loop on an item
        :param time_epoch: starting time for the loop on an epoch
        :param time_start: starting time of the entire process
        
        :return: 
        '''

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
        cdef int user_id = 0
        cdef int current_epoch
        cdef float error = 0.0
        cdef float prediction = 0.0
        cdef long time_item = 0
        cdef long time_epoch = 0
        cdef long time_start = 0

        print("\nStart the learnign process:")

        R_i_csc = self.urm_train.copy()
        R_i = self.urm_train.tocsr()

        self.R_i_indices = R_i.indices
        self.R_i_indptr = R_i.indptr
        self.R_i_data = R_i.data

        time_start = time.time()

        for current_item in range(items):

            #print("\n\nITEM {}\n\n".format(current_item))
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

                for sample_index in range(len(ri_indices)):

                    user_id = ri_indices[sample_index]

                    R_iu_data = self.R_i_data[self.R_i_indptr[user_id]:self.R_i_indptr[user_id+1]]
                    R_iu_indices = self.R_i_indices[self.R_i_indptr[user_id]:self.R_i_indptr[user_id+1]]

                    prediction = 0.0

                    for index_inner in range(len(R_iu_indices)):

                        item_id = R_iu_indices[index_inner]

                        if Si_mask[item_id] == True and item_id != current_item:
                            prediction += R_iu_data[index_inner] * Si_dense[item_id]

                    error = prediction - ri_data[sample_index]

                    for index_inner in range(len(R_iu_indices)):

                        item_id = R_iu_indices[index_inner]

                        if Si_mask[item_id] == True:
                            Si_dense[item_id] -= error * R_iu_data[index_inner] * self.l + Si_dense[item_id] * self.b + self.g


                for index_inner in range(len(Si_indices)):
                    Si_data[index_inner] = Si_dense[Si_indices[index_inner]]


            self.S_data[self.S_indptr[current_item]:self.S_indptr[current_item + 1]:] = Si_data

            #print("Elapsed time item {}: {} s, sample/sec {:.2f}".format(current_item, time.time() - time_item, len(ri_indices)*self.epochs/(time.time() - time_item)))


            for index in range(len(Si_indices)):

                item_id = Si_indices[index]

                Si_mask[item_id] = False
                Si_dense[item_id] = 0.0


            #print("Elapsed time item {}: {}".format(current_item, time.time() - time_item))

            if current_item != 0 and (current_item % 10000==0 or current_item == items):

                itemPerSec = current_item/(time.time()-time_start)
                print("Processed {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                    current_item, current_item*1.0/items*100, itemPerSec, (time.time()-time_start)/60))

                sys.stdout.flush()
                sys.stderr.flush()

        if current_item != items:
            itemPerSec = items/(time.time()-time_start)
            print("Processed {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                        items, items*1.0/items*100, itemPerSec, (time.time()-time_start)/60))



        return sps.csc_matrix((self.S_data, self.S_indices, self.S_indptr), shape=(items,items))


    def fit(self, topK = 100, shrink=100, normalize = True, mode = "cosine"):

        print("SLIM ElasticNet with structure, beginning of the fit process.\n")

        sim_comp = Cosine_Similarity(self.icm, topK, shrink, normalize, mode)
        self.S = sim_comp.compute_similarity()
        self.S_indices = self.S.indices
        self.S_data = self.S.data
        self.S_indptr = self.S.indptr

        return SLIM_Elastic_Net_Structured_Cython.learning_process(self)
