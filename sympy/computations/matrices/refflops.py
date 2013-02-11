
prod = lambda *args: reduce(lambda x, y: x*y, args, 1)

axpy = lambda alpha, X, Y: X.shape[0]*X.shape[1]

gemm = lambda alpha, A, B, beta, C: A.shape[0] * A.shape[1] * B.shape[1]
symm = lambda alpha, A, B, beta, C: A.shape[0] * A.shape[1] * B.shape[1]
trmm = lambda alpha, A, B: A.shape[0] * A.shape[1] * B.shape[1]

gemv = lambda alpha, A, X, Y: A.shape[0] * A.shape[1]
trsv = lambda A, X: A.shape[0]**2

laswp = lambda PA, ipiv: PA.shape[0] * PA.shape[1]

def getrf(A):
    m, n = A.shape
    return m*n**2 - 1/3*n**3 - 1/2*n**2 + 5/6*n
def getrs(A, B):
    n = A.shape[0]
    return 2*n**2 - n

gesv = lambda A, B: getrf(A) + getrs(A, B)

def potrf(A):
    n = A.shape[0]
    return 1/3*n**3 + 1/2*n**2 + 1/6*n
def potrs(A, B):
    n = A.shape[0]
    return 2*n**2

posv = lambda A, B: potrf(A) + potrs(A, B)
