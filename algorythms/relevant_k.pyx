'''
@author: Luca Vavassori
'''

import numpy as np
cimport numpy as np
from scipy import sparse as sps


# EXTRACTION OF RELEVANT K FOR EACH ITEM

cdef np.ndarray relevant_k(object icm, long i):

    col_i = icm[:,i]
    result = col_i.T.dot(icm)
    return result.indices


cpdef relevant_k_matrix(icm):

    cdef int rows, cols
    rows, cols = icm.shape
    cdef np.ndarray[int, ndim=1, mode='c'] indptrR
    dataR, indicesR = [], []
    indptrR = np.zeros(cols+1, dtype=np.int32)

    cdef int i = 0
    cdef np.ndarray[int, ndim=1, mode='c'] tempD
    cdef np.ndarray[int, ndim=1, mode='c'] tempI

    for i in range(cols):
        tempI = relevant_k(icm, i)
        tempD = np.ones((tempI.shape[0]), dtype=np.int32)
        dataR = np.append(dataR, np.asarray(tempD))
        indicesR = np.append(indicesR, np.asarray(tempI))
        indptrR[i+1] = indptrR[i] + tempD.shape[0]

    return sps.csc_matrix((dataR,indicesR,np.asarray(indptrR)), shape=(rows,cols))























