from sympy import Symbol
from sympy.matrices.expressions import MatrixSymbol, BlockMatrix
from sympy.computations.inplace import inplace_compile
from sympy.computations.matrices.compile import compile
from sympy.computations.matrices.blocks import JoinBlocks
from sympy.computations.matrices.fortran2 import *

def test_DAG_search():
    A,B,C,D = [MatrixSymbol(a, 3, 3) for a in 'ABCD']
    X = BlockMatrix([[A, B], [C, D]])
    assert next(compile([A,B,C,D], [X])) == JoinBlocks(X)

