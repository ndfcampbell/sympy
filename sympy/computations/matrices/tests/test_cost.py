from sympy.computations.matrices.cost import (flops, numelements,
        nbytesofoutputs, nbytesofinputs, commcost, memtime, compcost)
from sympy import MatrixSymbol, Symbol, S
from sympy.computations.matrices.blas import GEMM

def test_memtime():
    memhierarchy = [(1, 2), (.1, 4), (.01, 100000)]
    assert memtime(1, memhierarchy) == 1
    assert memtime(3, memhierarchy) == 12
    assert memtime(5, memhierarchy) == 32
    assert memtime(15, memhierarchy) == 942

def test_numelements():
    assert numelements(Symbol('x') + 3) == 1
    assert numelements(MatrixSymbol('x', 2, 3) * 2) == 6

A = MatrixSymbol('A', 3, 4)
B = MatrixSymbol('B', 4, 5)
C = MatrixSymbol('C', 3, 5)
comp = GEMM(S(13), A, B, S(14), C, 'D')
scomp = GEMM(S(13), A, B, S(14), C, 'S')

def test_nbytesoutputs():
    assert nbytesofoutputs(comp) == 3*5 * 8
    assert nbytesofoutputs(scomp) == 3*5 * 4

def test_nbytesinputs():
    assert nbytesofinputs(comp) == (3*4 + 4*5 + 3*5 + 1 + 1) * 8
    assert nbytesofinputs(scomp) == (3*4 + 4*5 + 3*5 + 1 + 1) * 4
