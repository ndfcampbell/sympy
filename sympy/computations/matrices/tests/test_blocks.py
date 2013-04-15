from sympy import Symbol
from sympy.matrices.expressions import (MatrixSymbol, BlockMatrix, blockcut,
        block_collapse)
from sympy.computations.inplace import inplace_compile
from sympy.computations.matrices.compile import compile
from sympy.computations.matrices.blocks import JoinBlocks
from sympy.computations.matrices.fortran2 import *

def test_DAG_search():
    A,B,C,D = [MatrixSymbol(a, 3, 3) for a in 'ABCD']
    X = BlockMatrix([[A, B], [C, D]])
    assert next(compile([A,B,C,D], [X])) == JoinBlocks(X)

def test_blockcut():
    n = 1024
    X = MatrixSymbol('X', n, n)
    XX = blockcut(X, (n//2, n//2), (n//2, n//2))
    assert next(compile([X], [XX]))

def test_block_matrixmultiply():
    n = 1024
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', n, n)
    XX = blockcut(X, (n//2, n//2), (n//2, n//2))
    YY = blockcut(Y, (n//2, n//2), (n//2, n//2))
    out = block_collapse(XX*YY)
    assert next(compile([X, Y], [out]))
