import numpy as np
import scipy.sparse
import scipy
from sklearn.model_selection import train_test_split


urm = scipy.sparse.load_npz("/home/luca/PycharmProjects/rec-sys-project/files/urm.npz")

X_train, X_test = train_test_split(urm, test_size=0.2)
print("splittato")

X_train = scipy.sparse.csc_matrix(X_train)
X_test = scipy.sparse.csc_matrix(X_test)

scipy.sparse.save_npz("/home/luca/PycharmProjects/rec-sys-project/files/train.npz", X_train)
scipy.sparse.save_npz("/home/luca/PycharmProjects/rec-sys-project/files/test.npz", X_test)

print(X_train)
print(X_test)