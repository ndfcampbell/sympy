from sympy.computations.matrices.blas import GEMM, SYMM
from sympy.computations.matrices.shared import (alpha, beta, n, m, k, A, B, C,
        x, a, b)
from sympy import Q, S

# pattern is (source expression, target expression, wilds, condition)
blas_patterns = [
    (GEMM._outputs[0], GEMM(*GEMM._inputs), GEMM._inputs, GEMM.condition),
    (alpha*A*B, GEMM(alpha, A, B, S.Zero, B), (alpha, A, B), True),
    (SYMM._outputs[0], SYMM(*SYMM._inputs), SYMM._inputs, SYMM.condition),
    (alpha*A*B, SYMM(alpha, A, B, S.Zero, B), (alpha, A, B), SYMM.condition)
]

patterns = blas_patterns
