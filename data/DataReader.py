#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/01/2018

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import scipy.sparse as sps
import pickle


def split_big_CSR_in_columns(sparse_matrix_to_split, num_split = 2):
    """
    The function returns a list of split for the given matrix
    :param sparse_matrix_to_split:
    :param num_split:
    :return:
    """

    if num_split<1 or num_split > sparse_matrix_to_split.shape[1]:
        raise ValueError("split_big_CSR_in_columns: num_split parameter not valid, value must be between 1 and {}, provided was {}".format(
            sparse_matrix_to_split.shape[1], num_split))


    if num_split == 1:
        return [sparse_matrix_to_split]



    n_column_split = int(sparse_matrix_to_split.shape[1]/num_split)

    sparse_matrix_split_list = []

    for num_current_split in range(num_split):

        start_col = n_column_split*num_current_split

        if num_current_split +1 == num_split:
            end_col = sparse_matrix_to_split.shape[1]
        else:
            end_col = n_column_split*(num_current_split + 1)

        print("split_big_CSR_in_columns: Split {}, columns: {}-{}".format(num_current_split, start_col, end_col))

        sparse_matrix_split_list.append(sparse_matrix_to_split[:,start_col:end_col])

    return sparse_matrix_split_list





def loadCSVintoSparse (filePath, header = False, separator="::"):

    values, rows, cols = [], [], []

    fileHandle = open(filePath, "r")
    numCells = 0

    if header:
        fileHandle.readline()

    for line in fileHandle:
        numCells += 1
        if (numCells % 1000000 == 0):
            print("Processed {} cells".format(numCells))

        if (len(line)) > 1:
            line = line.split(separator)

            line[-1] = line[-1].replace("\n", "")

            if not line[2] == "0" and not line[2] == "NaN":
                rows.append(int(line[0]))
                cols.append(int(line[1]))
                values.append(float(line[2]))

    fileHandle.close()

    return  sps.csr_matrix((values, (rows, cols)), dtype=np.float32)







def removeZeroRatingRowAndCol(URM, ICM = None):

    URM = check_matrix(URM, "csr")
    rows = URM.indptr
    numRatings = np.ediff1d(rows)
    user_mask = numRatings >= 1

    URM = URM[user_mask,:]

    cols = URM.tocsc().indptr
    numRatings = np.ediff1d(cols)
    item_mask = numRatings >= 1

    URM = URM[:,item_mask]

    removedUsers = np.arange(0, len(user_mask))[np.logical_not(user_mask)]
    removedItems = np.arange(0, len(item_mask))[np.logical_not(item_mask)]

    if ICM is not None:

        ICM = ICM[item_mask,:]

        return URM.tocsr(), ICM.tocsr(), removedUsers, removedItems


    return URM.tocsr(), removedUsers, removedItems




def splitTrainTestValidation(URM_all, splitProbability = list([0.6, 0.2, 0.2])):

    URM_all = URM_all.tocoo()
    shape = URM_all.shape

    numInteractions= len(URM_all.data)

    split = np.random.choice([1, 2, 3], numInteractions, p=splitProbability)


    trainMask = split == 1
    URM_train = sps.coo_matrix((URM_all.data[trainMask], (URM_all.row[trainMask], URM_all.col[trainMask])), shape = shape)
    URM_train = URM_train.tocsr()

    testMask = split == 2

    URM_test = sps.coo_matrix((URM_all.data[testMask], (URM_all.row[testMask], URM_all.col[testMask])), shape = shape)
    URM_test = URM_test.tocsr()

    validationMask = split == 3

    URM_validation = sps.coo_matrix((URM_all.data[validationMask], (URM_all.row[validationMask], URM_all.col[validationMask])), shape = shape)
    URM_validation = URM_validation.tocsr()

    return URM_train, URM_test, URM_validation


import time, sys

def urllretrieve_reporthook(count, block_size, total_size):

    global start_time

    if count == 0:
        start_time = time.time()
        return

    duration = time.time() - start_time + 1

    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count*block_size*100/total_size),100)

    sys.stdout.write("\rDataReader: Downloaded {:.2f}%, {:.2f} MB, {:.0f} KB/s, {:.0f} seconds passed".format(
                    percent, progress_size / (1024 * 1024), speed, duration))

    sys.stdout.flush()




from Base.Recommender_utils import check_matrix

def removeFeatures(ICM, minOccurrence = 5, maxPercOccurrence = 0.30, reconcile_mapper = None):
    """
    The function eliminates the values associated to feature occurring in less than the minimal percentage of items
    or more then the max. Shape of ICM is reduced deleting features.
    :param ICM:
    :param minPercOccurrence:
    :param maxPercOccurrence:
    :param reconcile_mapper: DICT mapper [token] -> index
    :return: ICM
    :return: deletedFeatures
    :return: DICT mapper [token] -> index
    """

    ICM = check_matrix(ICM, 'csc')

    n_items = ICM.shape[0]

    cols = ICM.indptr
    numOccurrences = np.ediff1d(cols)

    feature_mask = np.logical_and(numOccurrences >= minOccurrence, numOccurrences <= n_items*maxPercOccurrence)

    ICM = ICM[:,feature_mask]

    deletedFeatures = np.arange(0, len(feature_mask))[np.logical_not(feature_mask)]

    print("RemoveFeatures: removed {} features with less then {} occurrencies, removed {} features with more than {} occurrencies".format(
        sum(numOccurrences < minOccurrence), minOccurrence,
        sum(numOccurrences > n_items*maxPercOccurrence), int(n_items*maxPercOccurrence)
    ))

    if reconcile_mapper is not None:
        reconcile_mapper = reconcile_mapper_with_removed_tokens(reconcile_mapper, deletedFeatures)

        return ICM, deletedFeatures, reconcile_mapper


    return ICM, deletedFeatures

def reconcile_mapper_with_removed_tokens(mapper_dict, indices_to_remove):
    """

    :param mapper_dict: must be a mapper of [token] -> index
    :param indices_to_remove:
    :return:
    """

    # When an index has to be removed:
    # - Delete the corresponding key
    # - Decrement all greater indices

    indices_to_remove = set(indices_to_remove)
    removed_indices = []

    # Copy key set
    dict_keys = list(mapper_dict.keys())

    # Step 1, delete all values
    for key in dict_keys:

        if mapper_dict[key] in indices_to_remove:

            removed_indices.append(mapper_dict[key])
            del mapper_dict[key]


    removed_indices = np.array(removed_indices)


    # Step 2, decrement all remaining indices to fill gaps
    # Every index has to be decremented by the number of deleted tokens with lower index
    for key in mapper_dict.keys():

        lower_index_elements = np.sum(removed_indices<mapper_dict[key])
        mapper_dict[key] -= lower_index_elements


    return mapper_dict



#from data.URM_Dense_K_Cores import select_k_cores



class DataReader(object):
    """
    Abstract class for the DataReaders, each shoud be implemented for a specific dataset
    DataReaders provide two functionalrity
    - They load the data from the "original" subfolder
    - If the sparse matrices are not found, it loads the data (and saves it in the "original" subfolder), from the
        dataset-specific data structures
    """

    # This subfolder contains the preprocessed data, already loaded from the original data file
    DATASET_SUBFOLDER_ORIGINAL = "original/"

    # Available URM split
    AVAILABLE_URM = ["URM_train", "URM_validation", "URM_test"]

    # Available ICM for the given dataset, there might be no ICM, one or many
    AVAILABLE_ICM = []

    # Mappers existing for all datasets, associating USER_ID and ITEM_ID to the new designation
    GLOBAL_MAPPER = ["item_original_ID_to_index", "user_original_ID_to_index"]

    # Mappers specific for a given dataset, they might be related to more complex data structures or FEATURE_TOKENs
    DATASET_SPECIFIC_MAPPER = []


    def __init__(self, apply_k_cores = None):

        super(DataReader, self).__init__()

        self.apply_k_cores = apply_k_cores

        # NOTE: the presence of K-core will influence the file name but not the attribute name
        if self.apply_k_cores is None or self.apply_k_cores == 1:
            self.k_cores_value = 1
            self.k_cores_name_suffix = ""
        else:
            self.k_cores_value = self.apply_k_cores
            self.k_cores_name_suffix = "_{}-cores".format(self.apply_k_cores)

        if self.apply_k_cores is not None and self.apply_k_cores <= 0:
            raise ValueError("DataSplitter: apply_k_cores can only be either a positive number >= 1 or None. Provided value was '{}'".format(self.apply_k_cores))


        self.item_original_ID_to_index = {}
        self.item_index_to_original_ID = {}

        self.user_original_ID_to_index = {}
        self.user_index_to_original_ID = {}

        try:

            self._load_preprocessed_data()
            return

        except (FileNotFoundError, EOFError):

            # Sparse matrices in "original" subfolder not found, reading from original data structure
            self.load_from_original_file()


    def save_mappers(self):
        """
        Saves the mappers for the given dataset. Mappers associate the original ID of user, item, feature, to the
        index in the sparse matrix
        :param dataset_specific_mappers_list:
        :return:
        """

        mappers_list = list(self.GLOBAL_MAPPER)
        mappers_list.extend(self.DATASET_SPECIFIC_MAPPER)

        for ICM_name in self.AVAILABLE_ICM:
            mappers_list.append("tokenToFeatureMapper_{}".format(ICM_name))

        #mappers_list.extend(self.DATASET_SPECIFIC_MAPPER)

        for mapper_name in mappers_list:
            mapper_data = self.__getattribute__(mapper_name)
            pickle.dump(mapper_data, open(self.data_path + mapper_name + self.k_cores_name_suffix, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


    def load_mappers(self, k_cores_name_suffix = None):
        """
        Loads all saved mappers for the given dataset. Mappers are the union of GLOBAL mappers and dataset specific ones
        :return:
        """

        if k_cores_name_suffix is None:
            k_cores_name_suffix = self.k_cores_name_suffix

        mappers_list = list(self.GLOBAL_MAPPER)
        mappers_list.extend(self.DATASET_SPECIFIC_MAPPER)

        for ICM_name in self.AVAILABLE_ICM:
            mappers_list.append("tokenToFeatureMapper_{}".format(ICM_name))
        #mappers_list.extend(self.DATASET_SPECIFIC_MAPPER)

        for mapper_name in mappers_list:
            self.__setattr__(mapper_name, pickle.load(open(self.data_path + mapper_name + k_cores_name_suffix, "rb")))




    def load_from_original_file(self):
        raise NotImplementedError("DataReader: load_from_original_file not implemented for the chosen DataReader")


    def _load_preprocessed_data(self, splitSubfolder = DATASET_SUBFOLDER_ORIGINAL, ICM_to_load = None):
        """
        Loads ICM and URM from "original" subfolder
        :param splitSubfolder:
        :param ICM_to_load:
        :return:
        """

        # Try to load with required K core
        try:

            self._load_preprocessed_data_with_given_k_core(splitSubfolder = splitSubfolder,
                                                           ICM_to_load = ICM_to_load,
                                                           k_cores_name_suffix = self.k_cores_name_suffix)

        except FileNotFoundError as splitNotFoundException:

            # If not found, try to load  zero-core
            try:

                self._load_preprocessed_data_with_given_k_core(splitSubfolder = splitSubfolder,
                                                               ICM_to_load = ICM_to_load,
                                                               k_cores_name_suffix = "")


            except FileNotFoundError as splitNotFoundException:

                # No data was preprocessed
                raise splitNotFoundException


        self.n_items = self.URM_all.shape[1]

        '''# Apply required K - core on zero-core data from ORIGINAL split
        self.URM_all, removedUsers, removedItems = select_k_cores(self.URM_all, k_value = self.k_cores_value, reshape=True)

        self.item_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.item_original_ID_to_index, removedItems)
        self.user_original_ID_to_index = reconcile_mapper_with_removed_tokens(self.user_original_ID_to_index, removedUsers)

        print("DataReader: Removed {} users and {} items with less than {} interactions".format(len(removedUsers), len(removedItems), self.k_cores_value))

        ICM_filter_mask = np.ones(self.n_items, dtype=np.bool)
        ICM_filter_mask[removedItems] = False

        self.n_items = self.URM_all.shape[1]
        self.n_users = self.URM_all.shape[0]

        print("DataReader: Removing items from ICMs... ")'''


        if ICM_to_load is None:
            ICM_to_reconcile_list = self.AVAILABLE_ICM.copy()

        else:
            ICM_to_reconcile_list = [ICM_to_load]


        for ICM_name in ICM_to_reconcile_list:

            print("DataReader: Removing items from ICMs... {}".format(ICM_name))

            ICM_object = getattr(self, ICM_name)
            ICM_object = ICM_object[ICM_filter_mask,:]

            ICM_mapper_name = "tokenToFeatureMapper_{}".format(ICM_name)
            ICM_mapper_object = getattr(self, ICM_mapper_name)

            ICM_object, _, ICM_mapper_object = removeFeatures(ICM_object, minOccurrence = 1, maxPercOccurrence = 1.00,
                                                                                   reconcile_mapper = ICM_mapper_object)

            setattr(self, ICM_name, ICM_object)
            setattr(self, ICM_mapper_name, ICM_mapper_object)



        print("DataReader: Removing items from ICMs... done")



    def _load_preprocessed_data_with_given_k_core(self, splitSubfolder = DATASET_SUBFOLDER_ORIGINAL, ICM_to_load = None, k_cores_name_suffix = None):

        if ICM_to_load not in self.AVAILABLE_ICM and ICM_to_load is not None:
            raise ValueError("DataReader: ICM to load not recognized. Available values are {}, passed was '{}'".format(self.AVAILABLE_ICM, ICM_to_load))

        if k_cores_name_suffix is None:
            k_cores_name_suffix = self.k_cores_name_suffix

        self.ICM_to_load = ICM_to_load


        print("DataReader: loading data...")

        self.data_path = "./data/" + self.DATASET_SUBFOLDER + splitSubfolder

        try:
            if splitSubfolder == DataReader.DATASET_SUBFOLDER_ORIGINAL:
                self.URM_all = sps.load_npz(self.data_path + "URM_all{}.npz".format(k_cores_name_suffix))

            else:
                for URM_name in self.AVAILABLE_URM:
                    setattr(self, URM_name, sps.load_npz(self.data_path + "{}{}.npz".format(URM_name, k_cores_name_suffix)))



            if ICM_to_load is None:

                for ICM_name in self.AVAILABLE_ICM:
                    setattr(self, ICM_name, sps.load_npz(self.data_path + "{}{}.npz".format(ICM_name, k_cores_name_suffix)))

            else:
                setattr(self, ICM_to_load, sps.load_npz(self.data_path + "{}{}.npz".format(ICM_to_load, k_cores_name_suffix)))


            self.load_mappers(k_cores_name_suffix = k_cores_name_suffix)

            print("DataReader: loading {} complete".format(k_cores_name_suffix))


        except FileNotFoundError as splitNotFoundException:

            print("DataReader: URM or ICM {} not found".format(k_cores_name_suffix))

            raise splitNotFoundException



    def _merge_ICM(self, ICM1, ICM2, mapper_ICM1, mapper_ICM2):

        ICM_all = sps.hstack([ICM1, ICM2], format='csr')

        mapper_ICM_all = mapper_ICM1.copy()

        for key in mapper_ICM2.keys():
            mapper_ICM_all[key] = mapper_ICM2[key] + len(mapper_ICM1)

        return  ICM_all, mapper_ICM_all




    def _get_item_index(self, item_id, if_new = "add"):
        """
        From the id in the input files returns the index to be used in the sparse matrix
        :param item_id:
        :return:
        """

        if item_id in self.item_original_ID_to_index:
            itemIndex = self.item_original_ID_to_index[item_id]

        elif if_new == "add":
            itemIndex = len(self.item_original_ID_to_index)
            self.item_original_ID_to_index[item_id] = itemIndex
            self.item_index_to_original_ID[itemIndex] = item_id

            self.n_items = len(self.item_original_ID_to_index)

        elif if_new == "exception":
            # force raise exception
            self.item_original_ID_to_index[item_id]

        return itemIndex


    def _get_user_index(self, user_id, if_new = "add"):
        """
        From the id in the input files returns the index to be used in the sparse matrix
        :param user_id:
        :return:
        """

        if user_id in self.user_original_ID_to_index:
            userIndex = self.user_original_ID_to_index[user_id]

        elif if_new == "add":
            userIndex = len(self.user_original_ID_to_index)
            self.user_original_ID_to_index[user_id] = userIndex
            self.user_index_to_original_ID[userIndex] = user_id

            self.n_users = len(self.user_original_ID_to_index)

        elif if_new == "exception":
            # force raise exception
            self.user_original_ID_to_index[user_id]

        return userIndex



    def _get_token_index_from_mapper(self, token_original_ID_to_index, token_id, if_new = "add", token_index_to_original_ID = None):
        """
        From the id in the input files returns the index to be used in the sparse matrix
        :param user_id:
        :return:
        """

        if token_id in token_original_ID_to_index:
            token_index = token_original_ID_to_index[token_id]

        elif if_new == "add":
            token_index = len(token_original_ID_to_index)
            token_original_ID_to_index[token_id] = token_index

            if token_index_to_original_ID is not None:
                token_index_to_original_ID[token_index] = token_id

        elif if_new == "exception":
            # force raise exception
            token_original_ID_to_index[token_id]

        return token_index







    def _get_token_index(self, token, tokenToFeatureMapper, addIfNew = True):
        """
        From the id in the input files returns the index to be used in the sparse matrix
        :param token:
        :return: token_index
        """

        if token in tokenToFeatureMapper:
            token_index = tokenToFeatureMapper[token]

        elif addIfNew:
            token_index = len(tokenToFeatureMapper)
            tokenToFeatureMapper[token] = token_index

        else:
            # force raise exception
            tokenToFeatureMapper[token]

        return token_index


    def get_ICM(self):

        if self.ICM_to_load is not None:

            return getattr(self, self.ICM_to_load).copy()

        elif len(self.AVAILABLE_ICM) != 0:

            return getattr(self, self.AVAILABLE_ICM[0]).copy()

        else:
            return None



    def get_ICM_all_available(self):


        if self.ICM_to_load is None:

            result = []

            for ICM_name in self.AVAILABLE_ICM:

                result.append(getattr(self, ICM_name).copy())

            return result

        else:
            return [self.ICM.copy()]



    def get_URM_train(self):
        return self.URM_train.copy()

    def get_URM_test(self):
        return self.URM_test.copy()

    def get_URM_validation(self):
        return self.URM_validation.copy()

    #
    # def get_URM_train(self):
    #     raise NotImplementedError("DataReader: get_URM_train not implemented")
    #
    # def get_URM_test(self):
    #     raise NotImplementedError("DataReader: get_URM_test not implemented")
    #
    # def get_URM_validation(self):
    #     raise NotImplementedError("DataReader: get_URM_validation not implemented")
    #
    # def get_ICM(self):
    #     raise NotImplementedError("DataReader: get_ICM not implemented")
    #
    # def get_ICM_all_available(self):
    #     raise NotImplementedError("DataReader: get_ICM not implemented")


    def get_hyperparameters_for_rec_class(self, target_recommender):

        print("DataReader: No optimal parameters available for this dataset.")

        return {}


    def get_model_for_rec_class(self, target_recommender):

        print("DataReader: No model available for this dataset.")

        return None



    def get_statistics(self):

        n_items = self.URM_train.shape[1]
        n_users = self.URM_train.shape[0]

        print("DataReader: current dataset is: {}\n"
              "\tNumber of items: {}\n"
              "\tNumber of users: {}\n"
              "\tNumber of interactions in train: {}\n"
              "\tTrain density: {:.2E}\n"
              "\tNumber of interactions in validation: {}\n"
              "\tValidation density: {:.2E}".format(
            self.__class__, n_items, n_users,
            self.URM_train.nnz, self.URM_train.nnz/(n_items*n_users),
            self.URM_validation.nnz, self.URM_validation.nnz/(n_items*n_users)))


    def downloadFromURL(self, URL, destinationFolder):

        from urllib.request import urlretrieve

        urlretrieve (URL, destinationFolder, reporthook=urllretrieve_reporthook)

        sys.stdout.write("\n")
        sys.stdout.flush()




    def loadCSVintoSparse_mapID (self, filePath, header = False, separator=",", if_new_user = "add", if_new_item = "add"):

        if if_new_user not in ["add", "ignore", "exception"]:
            raise ValueError("DataReader: if_new_user parameter not recognized. Accepted values are 'add', 'ignore', 'exception', provided was '{}'".format(if_new_user))

        if if_new_item not in ["add", "ignore", "exception"]:
            raise ValueError("DataReader: if_new_item parameter not recognized. Accepted values are 'add', 'ignore', 'exception', provided was '{}'".format(if_new_item))

        if if_new_user == "ignore":
            if_new_user_get_user_index = "exception"
        else:
            if_new_user_get_user_index = if_new_user

        if if_new_item == "ignore":
            if_new_item_get_item_index = "exception"
        else:
            if_new_item_get_item_index = if_new_item




        # Use array as it requires MUCH less space than lists
        dataBlock = 10000000

        values = np.zeros(dataBlock, dtype=np.float64)
        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)

        numCells = 0

        fileHandle = open(filePath, "r")


        if header:
            fileHandle.readline()

        for line in fileHandle:

            if (numCells % 1000000 == 0 and numCells!=0):
                print("Processed {} cells".format(numCells))

            if (len(line)) > 1:
                line = line.split(separator)

                line[-1] = line[-1].replace("\n", "")

                # Do not transfor userID and itemID into integers to support
                # alphanumeric IDs
                try:

                    userIndex = self._get_user_index(line[0], if_new = if_new_user_get_user_index)
                    movieIndex = self._get_item_index(line[1], if_new = if_new_item_get_item_index)


                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.float64)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.int32)))


                    rows[numCells] = userIndex
                    cols[numCells] = movieIndex
                    values[numCells] = float(line[2])

                    numCells += 1

                    #
                    # rows.append(userIndex)
                    # cols.append(movieIndex)
                    # values.append(float(line[2]))

                except KeyError:
                    pass


        fileHandle.close()

        return  sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), dtype=np.float32)

