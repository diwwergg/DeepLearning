import numpy as np
import numba as nb
import time
from numba import cuda

# print(cuda.detect())
Acpu = np.random.rand(2, 2)
Agpu = cuda.to_device(Acpu)
A = Agpu.copy_to_host()

# this is a naive implementation of matrix multiplication
# big O is O(n^3)
def matmul(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C
# print(matmul(A, B))


# use cuda to accelerate the matrix multiplication
@cuda.jit
def cumm(A, B, C):
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = 0
        for k in range(A.shape[1]):
            C[i, j] += A[i, k] * B[k, j]

A = np.random.rand(300, 300)
B = np.random.rand(300, 300)
C = np.zeros((A.shape[0], B.shape[1])) # 256 * 256

Agpu = cuda.to_device(A)
Bgpu = cuda.to_device(B)
Cgpu = cuda.to_device(C)


threadperblock = (16, 16)
blockpergrid = tuple(
    int(np.ceil(C.shape[i]/t))
    for i, t in enumerate(threadperblock)
)
# print(blockpergrid)

# cpu
t1 = time.time()
matmul(A, B)
print(time.time() - t1)

# gpu
t1 = time.time()
# call def cumm
cumm[blockpergrid, threadperblock](Agpu, Bgpu, Cgpu)
print(time.time() - t1)


