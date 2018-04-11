'''
@author = Luca Vavassori, Alex Porciani
'''

import numpy as np
cimport numpy as np
from heapq import nlargest
cimport cython
from Base.Cython.cosine_similarity import Cosine_Similarity
from Base.metrics import roc_auc, precision, recall, rr, map, ndcg
import time
from libcpp cimport bool

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
    cdef bool exclude_seen


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

        for current_item in range(items):

            print("\n\nITEM {}\n\n".format(current_item))
            time_item = time.time()

            # Copy URM and set column i to 0, take the original column X
            R_i = self.urm_train.copy()
            ri = R_i[:, current_item].copy()
            R_i.data[R_i.indptr[current_item]:R_i.indptr[current_item + 1]] = 0
            R_i = R_i.tocsr()

            self.R_i_indices = R_i.indices
            self.R_i_indptr = R_i.indptr
            self.R_i_data = R_i.data


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

                        if Si_mask[item_id] == True:
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


    def fit(self, topK = 100, shrink=0, normalize = True, mode = "cosine"):

        sim_comp = Cosine_Similarity(self.icm, topK, shrink, normalize, mode)
        print("\nStart computing the relevant K items [ICM Similarity]\n")
        self.S = sim_comp.compute_similarity()
        self.S_indices = self.S.indices
        self.S_data = self.S.data
        self.S_indptr = self.S.indptr
        print("\nRelevant K computed [ICM Similarity]")
        SLIM_Elastic_Net_Cython.learning_process(self)


    def recommend(self, user_id, n=None, exclude_seen=False, filterTopPop = False, filterCustomItems = False):

        if n==None:
            n=self.urm_train.shape[1]-1

        # compute the scores using the dot product
        if self.sparse_weights:
            user_profile = self.urm_train[user_id]

            scores = user_profile.dot(self.S).toarray().ravel()    #sparse S

        else:

            user_profile = self.urm_train.indices[self.urm_train.indptr[user_id]:self.urm_train.indptr[user_id + 1]]
            user_ratings = self.urm_train.data[self.urm_train.indptr[user_id]:self.urm_train.indptr[user_id + 1]]

            relevant_weights = self.W[user_profile]
            scores = relevant_weights.T.dot(user_ratings)

        if exclude_seen:
            scores = self._filter_seen_on_scores(user_id, scores)

        if filterTopPop:
            scores = self._filter_TopPop_on_scores(scores)

        if filterCustomItems:
            scores = self._filterCustomItems_on_scores(scores)


        # rank items and mirror column to obtain a ranking in descending score
        #ranking = scores.argsort()
        #ranking = np.flip(ranking, axis=0)

        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index
        relevant_items_partition = (-scores).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]


        return ranking



    def evaluateRecommendations(self, URM_test_new, int at=5, int minRatingsPerUser=1, bool exclude_seen=True):
        """
        Speed info:
        - Sparse weighgs: batch mode is 2x faster than sequential
        - Dense weighgts: batch and sequential speed are equivalent


        :param URM_test_new:            URM to be used for testing
        :param at: 5                    Length of the recommended items
        :param minRatingsPerUser: 1     Users with less than this number of interactions will not be evaluated
        :param exclude_seen: True       Whether to remove already seen items from the recommended items

        :param mode: 'sequential', 'parallel', 'batch'
        :return:
        """

        cdef long nusers
        cdef int[:] rows
        cdef int[:] num_ratings

        self.urm_test = URM_test_new
        self.test_data = URM_test_new.data
        self.test_indices = URM_test_new.indices
        self.test_indptr = URM_test_new.indptr
        self.at = at
        self.min_ratings_per_user = minRatingsPerUser
        self.exclude_seen = exclude_seen

        nusers = self.urm_test.shape[0]

        # Prune users with an insufficient number of ratings
        rows = self.urm_test.indptr
        num_ratings = np.ediff1d(rows)
        for i in range(num_ratings):
            mask = i >= minRatingsPerUser
        users_to_evaluate = np.arange(nusers, dtype=np.int32)[mask]

        return SLIM_Elastic_Net_Cython.evaluateRecommendationsSequential(self, users_to_evaluate)


    cdef get_user_relevant_items(self, int user_id):

        return self.test_indices[self.test_indptr[user_id]:self.test_indptr[user_id + 1]]

    cdef get_user_test_ratings(self, int user_id):

        return self.test_data[self.test_indptr[user_id]:self.test_indptr[user_id + 1]]


    cdef evaluateRecommendationsSequential(self, int[:] users_to_evaluate):

        cdef long start_time = time.time()
        cdef float roc_auc_, precision_, recall_, map_, mrr_, ndcg_
        cdef int n_eval
        cdef int index, index_isin
        cdef int[:] recommended_items, relevant_items

        roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n_eval = 0

        for index in range(len(users_to_evaluate)):

            # Calling the 'evaluateOneUser' function instead of copying its code would be cleaner, but is 20% slower

            # Being the URM CSR, the indices are the non-zero column indexes
            relevant_items = self.get_user_relevant_items(users_to_evaluate[index])

            n_eval += 1

            recommended_items = SLIM_Elastic_Net_Cython.recommend(self ,users_to_evaluate[index], self.exclude_seen,
                                               self.at)

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            # evaluate the recommendation list with ranking metrics ONLY
            roc_auc_ += roc_auc(is_relevant)
            precision_ += precision(is_relevant)
            recall_ += recall(is_relevant, relevant_items)
            map_ += map(is_relevant, relevant_items)
            mrr_ += rr(is_relevant)
            ndcg_ += ndcg(recommended_items, relevant_items, relevance=self.get_user_test_ratings(users_to_evaluate[index]), at=self.at)



            if n_eval % 10000 == 0 or n_eval==len(users_to_evaluate)-1:
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Users per second: {:.0f}".format(
                                  n_eval,
                                  100.0* float(n_eval+1)/len(users_to_evaluate),
                                  time.time()-start_time,
                                  float(n_eval)/(time.time()-start_time)))




        if (n_eval > 0):
            roc_auc_ /= n_eval
            precision_ /= n_eval
            recall_ /= n_eval
            map_ /= n_eval
            mrr_ /= n_eval
            ndcg_ /= n_eval

        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {}

        results_run["AUC"] = roc_auc_
        results_run["precision"] = precision_
        results_run["recall"] = recall_
        results_run["map"] = map_
        results_run["NDCG"] = ndcg_
        results_run["MRR"] = mrr_

        return (results_run)


