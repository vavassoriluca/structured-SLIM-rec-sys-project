'''
@author = Luca Vavassori
'''

from scipy import  sparse as sps
from Base.Cython.cosine_similarity import Cosine_Similarity

x = sps.rand(10000,10000,0.01, format='csc')
print("Matrix generated! \n")
s = Cosine_Similarity(x)
print(s.compute_similarity())