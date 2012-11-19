from sympy.computations.matrices.core import MatrixCall, remove_numbers
from sympy.computations.core import unique
from sympy.computations.matrices.shared import detranspose
from sympy.computations.matrices.shared import (alpha, beta, n, m, k, A, B, C, S,
        x, a, b)
from sympy import Q

class BLAS(MatrixCall):
    """ Basic Linear Algebra Subroutine - Dense Matrix computation """
    flags = ["-lblas"]

class MM(BLAS):
    """ Matrix Multiply """
    _inputs   = (alpha, A, B, beta, C)
    _outputs  = (alpha*A*B + beta*C,)
    view_map  = {0: 4}
    condition = True

    @property
    def inputs(self):
        alpha, A, B, beta, C = self.raw_inputs
        coll = (alpha, detranspose(A), detranspose(B), beta, C)
        return tuple(unique(remove_numbers(coll)))

class GEMM(MM):
    """ General Matrix Multiply """
    pass

class SYMM(MM):
    """ Symmetric Matrix Multiply """
    condition = Q.symmetric(A) | Q.symmetric(B)

def test_valid():
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    C = MatrixSymbol('C', n, n)
    assert GEMM.valid((1, A, B, 2, C), True)
    assert not SYMM.valid((1, A, B, 2, C), True)
    assert SYMM.valid((1, A, B, 2, C), Q.symmetric(A))
    assert SYMM.valid((1, A, B, 2, C), Q.symmetric(B))
