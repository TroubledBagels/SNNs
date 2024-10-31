import numpy as np
import scipy as sp

# Generate random numbers
# np.random.seed(0)
# A = sp.sparse.random(10000, 10000, density=0.05, format='csr')
# # Remove under 0.5
# A.data[A.data < 0.5] = 0
# A.eliminate_zeros()
#

def cos_sim(_vec1, _vec2):
    numerator = _vec1.dot(_vec2.T)
    denominator = np.linalg.norm(_vec1) * np.linalg.norm(_vec2)
    if denominator == 0: return 0
    return numerator / denominator

arr_1 = np.array([0, 1, 2, 3, 4, 5])
arr_2 = np.array([1, 2, 3, 4, 5, 6])
arr_3 = np.array([0, 1, 2, 0, 3, 4, 5, 0, 0])
arr_4 = np.array([1, 2, 3, 0, 4, 5, 6, 0, 0])

print(cos_sim(arr_1, arr_2))
print(cos_sim(arr_3, arr_4))