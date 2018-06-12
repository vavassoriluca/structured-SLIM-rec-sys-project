#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import scipy.sparse as sps
import zipfile
from scipy.sparse import csc_matrix
import ast, csv, pickle

from data.DataReader import DataReader, removeFeatures, removeZeroRatingRowAndCol
#from data.URM_Dense_K_Cores import select_k_cores
#from Base.Recommender_utils import reshapeSparse
from data.DataReader import reconcile_mapper_with_removed_tokens


class TheMoviesDatasetReader(DataReader):

    DATASET_SUBFOLDER = "TheMoviesDataset/"
    AVAILABLE_ICM = ["ICM_all", "ICM_credits", "ICM_metadata"]
    DATASET_SPECIFIC_MAPPER = ["item_original_ID_to_title", "item_index_to_title"]

    EDITORIAL_ICM = "ICM_all"


    def __init__(self, apply_k_cores = None):
        """
        :param splitSubfolder:
        """

        super(TheMoviesDatasetReader, self).__init__()


    def load_from_original_file(self):
        # Load data from original

        print("TheMoviesDatasetReader: Loading original data")

        zipFile_path = "./data/" + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "the-movies-dataset.zip")

        except FileNotFoundError as fileNotFound:
            raise fileNotFound



        credits_path = dataFile.extract("credits.csv", path=zipFile_path + "the-movies-dataset/")
        metadata_path = dataFile.extract("movies_metadata.csv", path=zipFile_path + "the-movies-dataset/")
        movielens_tmdb_id_map_path = dataFile.extract("links.csv", path=zipFile_path + "the-movies-dataset/")


        self.tokenToFeatureMapper_ICM_credits = {}
        self.tokenToFeatureMapper_ICM_metadata = {}
        self.tokenToFeatureMapper_ICM_all = {}

        self.item_original_ID_to_title = {}
        self.item_index_to_title = {}

        print("TheMoviesDatasetReader: Loading ICM_credits")
        self.ICM_credits = self._loadICM_credits(credits_path, self.tokenToFeatureMapper_ICM_credits, header=True)

        print("TheMoviesDatasetReader: Loading ICM_metadata")
        self.ICM_metadata = self._loadICM_metadata(metadata_path, self.tokenToFeatureMapper_ICM_metadata, header=True)


        self.ICM_credits, _, self.tokenToFeatureMapper_ICM_credits = removeFeatures(self.ICM_credits, minOccurrence = 5, maxPercOccurrence = 0.30,
                                                                                    reconcile_mapper = self.tokenToFeatureMapper_ICM_credits)

        self.ICM_metadata, _, self.tokenToFeatureMapper_ICM_metadata = removeFeatures(self.ICM_metadata, minOccurrence = 5, maxPercOccurrence = 0.30,
                                                                                    reconcile_mapper = self.tokenToFeatureMapper_ICM_metadata)
        #self.ICM_credits = sps.csc_matrix((self.ICM_credits.data, self.ICM_credits.rows_indices, self.ICM_credits.cols_indptr),shape=(self.n_items, self.ICM_credits.shape[1]), dtype=np.float32)
        #self.ICM_credits = self.ICM_credits.tocsr()
        #self.tokenToFeatureMapper_credits = reconcile_mapper_with_removed_tokens(self.tokenToFeatureMapper_credits, removed_features_credits)
        #self.tokenToFeatureMapper_metadata = reconcile_mapper_with_removed_tokens(self.tokenToFeatureMapper_metadata, removed_features_metadata)

        shape = (self.n_items, self.ICM_credits.shape[1])
        self.ICM_credits = sps.csc_matrix((self.ICM_credits.data, self.ICM_credits.indices, self.ICM_credits.indptr), shape=shape, copy=True)

        # IMPORTANT: ICM uses TMDB indices, URM uses movielens indices
        # Load index mapper
        movielens_id_to_tmdb, tmdb_to_movielens_id = self._load_item_id_mappping(movielens_tmdb_id_map_path, header=True)

        # Modify saved mapper to accept movielens id instead of tmdb
        self._replace_tmdb_id_with_movielens(tmdb_to_movielens_id)


        print("TheMoviesDatasetReader: Loading URM")
        URM_path = dataFile.extract("ratings.csv", path=zipFile_path + "the-movies-dataset/")
        self.URM_all = self.loadCSVintoSparse_mapID (URM_path, header = True, separator=",", if_new_user = "add", if_new_item = "add")


        # Reconcile URM and ICM
        # Keep only items having ICM entries, remove all the others
        self.n_items = self.ICM_credits.shape[0]

        self.URM_all = self.URM_all[:,0:self.n_items]

        print(self.URM_all.shape)

        '''
        self.URM_all, removedUsers, removedItems = select_k_cores(self.URM_all, k_value = self.k_cores_value, reshape=True)
        self.URM_all, removedUsers, removedItems = removeZeroRatingRowAndCol(self.URM_all)

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)

        # Remove movie_ID discarded in previous step
        item_original_ID_to_title_old = self.item_original_ID_to_title.copy()

        for item_id in item_original_ID_to_title_old:

            if item_id not in self.item_original_ID_to_index:
                del self.item_original_ID_to_title[item_id]

        removed_item_mask = np.zeros(self.n_items, dtype=np.bool)
        removed_item_mask[removedItems] = True

        to_preserve_item_mask = np.logical_not(removed_item_mask)

        self.ICM_credits = self.ICM_credits[to_preserve_item_mask,:]
        self.ICM_metadata = self.ICM_metadata[to_preserve_item_mask,:]
        # URM is already clean

        self.n_items = self.ICM_credits.shape[0]

        '''
        self.ICM_all, self.tokenToFeatureMapper_ICM_all = self._merge_ICM(self.ICM_credits, self.ICM_metadata,
                                                                          self.tokenToFeatureMapper_ICM_credits,
                                                                          self.tokenToFeatureMapper_ICM_metadata)


        print("TheMoviesDatasetReader: saving URM_train and ICM")
        sps.save_npz("./data/TheMoviesDataset/URM_all.npz", self.URM_all)
        sps.save_npz("./data/TheMoviesDataset/ICM_all.npz", self.ICM_all)

        #self.save_mappers()


        print("TheMoviesDatasetReader: loading complete")



    def _load_item_id_mappping(self, movielens_tmdb_id_map_path, header=True):

        movielens_id_to_tmdb = {}
        tmdb_to_movielens_id = {}

        movielens_tmdb_id_map_file = open(movielens_tmdb_id_map_path, 'r', encoding="utf8")

        if header:
            movielens_tmdb_id_map_file.readline()


        for newMapping in movielens_tmdb_id_map_file:

            newMapping = newMapping.split(",")

            movielens_id = newMapping[0]
            tmdb_id = newMapping[2].replace("\n", "")

            movielens_id_to_tmdb[movielens_id] = tmdb_id
            tmdb_to_movielens_id[tmdb_id] = movielens_id


        return movielens_id_to_tmdb, tmdb_to_movielens_id



    def _replace_tmdb_id_with_movielens(self, tmdb_to_movielens_id):
        """
        Replace 'the original id' in such a way that it points to the same index
        :param tmdb_to_movielens_id:
        :return:
        """

        item_original_ID_to_index_movielens = {}
        item_index_to_original_ID_movielens = {}
        item_original_ID_to_title_movielens = {}

        # self.item_original_ID_to_index[item_id] = itemIndex
        # self.item_index_to_original_ID[itemIndex] = item_id

        for item_index in self.item_index_to_original_ID.keys():

            tmdb_id = self.item_index_to_original_ID[item_index]

            if tmdb_id in self.item_original_ID_to_title:
                movie_title = self.item_original_ID_to_title[tmdb_id]
            else:
                movie_title = ""

            movielens_id = tmdb_to_movielens_id[tmdb_id]

            item_index_to_original_ID_movielens[item_index] = movielens_id
            item_original_ID_to_index_movielens[movielens_id] = item_index
            item_original_ID_to_title_movielens[movielens_id] = movie_title


        # Replace the TMDB based mapper
        self.item_original_ID_to_index = item_original_ID_to_index_movielens
        self.item_index_to_original_ID = item_index_to_original_ID_movielens
        self.item_original_ID_to_title = item_original_ID_to_title_movielens







    def _loadICM_credits(self, credits_path, tokenToFeatureMapper_credits, header=True):


        values, rows, cols = [], [], []

        #parser_credits = parse_json(credits_path, header = header)
        numCells = 0

        credits_file = open(credits_path, 'r', encoding="utf8")

        if header:
            credits_file.readline()

        parser_credits = csv.reader(credits_file, delimiter=',', quotechar='"')


        for newCredits in parser_credits:

            # newCredits is a tuple of two strings, both are lists of dictionaries
            # {'cast_id': 14, 'character': 'Woody (voice)', 'credit_id': '52fe4284c3a36847f8024f95', 'gender': 2, 'id': 31, 'name': 'Tom Hanks', 'order': 0, 'profile_path': '/pQFoyx7rp09CJTAb932F2g8Nlho.jpg'}
            # {'cast_id': 14, 'character': 'Woody (voice)', 'credit_id': '52fe4284c3a36847f8024f95', 'gender': 2, 'id': 31, 'name': 'Tom Hanks', 'order': 0, 'profile_path': '/pQFoyx7rp09CJTAb932F2g8Nlho.jpg'}
            # NOTE: sometimes a dict value is ""Savannah 'Vannah' Jackson"", if the previous eval removes the commas "" "" then the parsing of the string will fail
            cast_list = []
            credits_list = []

            try:
                cast_list = ast.literal_eval(newCredits[0])
                credits_list = ast.literal_eval(newCredits[1])
            except Exception as e:
                print("TheMoviesDatasetReader: Exception while parsing: '{}', skipping".format(str(e)))


            movie_id = newCredits[2]

            movie_index = self._get_item_index(movie_id)

            cast_list.extend(credits_list)

            for cast_member in cast_list:
                name = cast_member["name"]

                numCells += 1
                if numCells % 100000 == 0:
                    print("Processed {} cells".format(numCells))

                feature_id = self._get_token_index(name, tokenToFeatureMapper_credits, addIfNew = True)

                # Rows movie ID
                # Cols features
                rows.append(movie_index)
                cols.append(feature_id)
                values.append(True)



        return sps.csr_matrix((values, (rows, cols)), dtype=np.bool)





    def _loadICM_metadata(self, metadata_path, tokenToFeatureMapper_metadata, header=True):


        values, rows, cols = [], [], []

        numCells = 0

        metadata_file = open(metadata_path, 'r', encoding="utf8")

        if header:
            metadata_file.readline()

        parser_metadata = csv.reader(metadata_file, delimiter=',', quotechar='"')


        for newMetadata in parser_metadata:

            numCells += 1
            if numCells % 100000 == 0:
                print("Processed {} cells".format(numCells))

            token_list = []

            if len(newMetadata) < 22:
                #Sono 6, ragionevole
                print("TheMoviesDatasetReader: Line too short, possible unwanted new line character, skipping")
                continue

            movie_id = newMetadata[5]
            movie_index = self._get_item_index(movie_id)


            if newMetadata[0] == "True":
                token_list.append("ADULTS_YES")
            else:
                token_list.append("ADULTS_NO")

            if newMetadata[1]:
                collection = ast.literal_eval(newMetadata[1])
                token_list.append("collection_" + str(collection["id"]))

            #budget = int(rating[2])

            if newMetadata[3]:
                genres = ast.literal_eval(newMetadata[3])

                for genre in genres:
                    token_list.append("genre_" + str(genre["id"]))


            orig_lang = newMetadata[7]
            title = newMetadata[8]

            if movie_id not in self.item_original_ID_to_title:
                self.item_original_ID_to_title[movie_id] = title

            if orig_lang:
                token_list.append("original_language_"+orig_lang)

            if newMetadata[12]:
                prod_companies = ast.literal_eval(newMetadata[12])
                for prod_company in prod_companies:
                    token_list.append("production_company_" + str(prod_company['id']))


            if newMetadata[13]:
                prod_countries = ast.literal_eval(newMetadata[13])
                for prod_country in prod_countries:
                    token_list.append("production_country_" + prod_country['iso_3166_1'])


            try:
                release_date = int(newMetadata[14].split("-")[0])
                token_list.append("release_date_" + str(release_date))
            except Exception:
                pass


            if newMetadata[17]:
                spoken_langs = ast.literal_eval(newMetadata[17])
                for spoken_lang in spoken_langs:
                    token_list.append("spoken_lang_" + spoken_lang['iso_639_1'])


            if newMetadata[18]:
                status = newMetadata[18]
                if status:
                    token_list.append("status_" + status)

            if newMetadata[21] == "True":
                token_list.append("VIDEO_YES")
            else:
                token_list.append("VIDEO_NO")


            for token in token_list:
                feature_id = self._get_token_index(token, tokenToFeatureMapper_metadata, addIfNew = True)

                rows.append(movie_index)
                cols.append(feature_id)
                values.append(True)


        return sps.csr_matrix((values, (rows, cols)), dtype=np.bool)





    def get_hyperparameters_for_rec_class(self, target_recommender):

        from KNN.item_knn_CBF import ItemKNNCBFRecommender
        from KNN.item_knn_CF import ItemKNNCFRecommender
        from KNN.user_knn_CF import UserKNNCFRecommender
        from SLIM_ElasticNet.SLIM_ElasticNet import MultiThreadSLIM_ElasticNet
        from SLIM_ElasticNet.SLIM_ElasticNet import SLIM_ElasticNet
        from GraphBased.P3alpha import P3alphaRecommender
        from GraphBased.RP3beta import RP3betaRecommender

        try:
            from MatrixFactorization.Cython.MF_BPR_Cython import MF_BPR_Cython
            from MatrixFactorization.MatrixFactorization_RMSE import FunkSVD
            from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
        except ImportError:
            MF_BPR_Cython = None
            FunkSVD = None
            SLIM_BPR_Cython = None


        hyperparam_dict = {}


        #
        # if target_recommender is ItemKNNCBFRecommender:
        #     hyperparam_dict["topK"] = 50
        #     hyperparam_dict["shrink"] = 100
        #     hyperparam_dict["similarity"] = 'jaccard'
        #     hyperparam_dict["normalize"] = True
        #
        #     return hyperparam_dict
        #
        # elif target_recommender is ItemKNNCFRecommender:
        #     hyperparam_dict["topK"] = 150
        #     hyperparam_dict["shrink"] = 0
        #     hyperparam_dict["similarity"] = 'cosine'
        #     hyperparam_dict["normalize"] = True
        #
        #     return hyperparam_dict
        #
        # elif target_recommender is UserKNNCFRecommender:
        #     hyperparam_dict["topK"] = 200
        #     hyperparam_dict["shrink"] = 0
        #     hyperparam_dict["similarity"] = 'jaccard'
        #     hyperparam_dict["normalize"] = True
        #
        #     return hyperparam_dict
        #
        # elif target_recommender is MF_BPR_Cython:
        #     hyperparam_dict["num_factors"] = 1
        #     hyperparam_dict["epochs"] = 11
        #     hyperparam_dict["batch_size"] = 1
        #     hyperparam_dict["learning_rate"] = 0.01
        #
        #     return hyperparam_dict
        #
        # elif target_recommender is FunkSVD:
        #     hyperparam_dict["num_factors"] = 1
        #     hyperparam_dict["epochs"] = 30
        #     hyperparam_dict["reg"] = 1e-5
        #     hyperparam_dict["learning_rate"] = 1e-4
        #
        #     return hyperparam_dict
        #
        # elif target_recommender is SLIM_BPR_Cython:
        #     hyperparam_dict["sgd_mode"] = 'adagrad'
        #     hyperparam_dict["epochs"] = 21
        #     hyperparam_dict["batch_size"] = 1
        #     hyperparam_dict["learning_rate"] = 0.1
        #     hyperparam_dict["topK"] = 200
        #
        #     return hyperparam_dict
        #
        # elif target_recommender is SLIM_ElasticNet or target_recommender is MultiThreadSLIM_RMSE:
        #     hyperparam_dict["topK"] = 200
        #     hyperparam_dict["positive_only"] = True
        #     hyperparam_dict["l1_penalty"] = 1e-5
        #     hyperparam_dict["l2_penalty"] = 1e-2
        #
        #     return hyperparam_dict
        #
        # elif target_recommender is P3alphaRecommender:
        #     hyperparam_dict["topK"] = 150
        #     hyperparam_dict["alpha"] = 1.3
        #     hyperparam_dict["normalize_similarity"] = True
        #
        #     return hyperparam_dict
        #
        #
        # elif target_recommender is RP3betaRecommender:
        #     hyperparam_dict["topK"] = 150
        #     hyperparam_dict["alpha"] = 0.9
        #     hyperparam_dict["beta"] = 0.6
        #     hyperparam_dict["normalize_similarity"] = True
        #
        #     return hyperparam_dict

        print("TheMoviesDatasetReader: No optimal parameters available for algorithm of class {}".format(target_recommender))

        return hyperparam_dict





TheMoviesDatasetReader()