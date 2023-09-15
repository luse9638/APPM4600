import numpy as np

## 2

# 2b)

# create matrices A and A^-1
A = 0.5 * np.array([[1, 1], [1 + 10 ** -10, 1 - 10 ** -10]])
Ainv = np.array([[1 - 10 ** 10, 10 ** 10], [1 + 10 ** 10, -1 * 10 ** 10]])

# calculate maximum and minimum singular values of A
maxSVDA = np.linalg.svd(A)[1][0] # sigma_1
minSVDA = np.linalg.svd(A)[1][1] # sigma_n

# condition number, k(A) = sigma_1 / sigma_n
condA = maxSVDA / minSVDA
print(condA)

