'''
@author = Luca Vavassori, Alex Porciani
'''

import numpy as np
cimport numpy as np
from scipy import sparse as sps
import datetime

from Base.Cython.cosine_similarity import Cosine_Similarity

cdef class Elastic_Net:

    cdef similarity
    cdef urm
    cdef icm
    cdef double l
    cdef double b
    cdef double g
    cdef int epochs

    def __init__(self, icm, urm, double l=0.01, double b=0.01, double g=0.01, int epochs=10):

        self.icm = icm
        self.urm = urm
        self.l = l
        self.b = b
        self.g = g
        self.epochs = epochs


    cdef learning_process(self):



        print("Start learnign process: ", datetime.datetime.now().time())



    def fit(self, topK = 100, shrink=0, normalize = True, mode = "cosine"):

        sim_comp = Cosine_Similarity(self.icm, topK, shrink, normalize, mode)
        print("Start computing the relevant K items [ICM Similarity]: ", datetime.datetime.now().time())
        self.similarity = sim_comp.compute_similarity()
        print("Relevant K computed [ICM Similarity]: ", datetime.datetime.now().time())
        Elastic_Net.learning_process(self)


