from sympy.matrices.expressions.blockmatrix import (block_symbol, BlockMatrix,
        block_cut)
from sympy.matrices.expressions import MatrixSymbol, MatrixExpr
from sympy import symbols

ms = MatrixSymbol

n, m, l, k = symbols('n m l k')

A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, k)
C = MatrixSymbol('C', k, l)
x = MatrixSymbol('x', m, 1)
y = MatrixSymbol('y', 1, n)


def test_symbol():
    result = block_symbol(A, {n: [n/2, n/2]})
    assert isinstance(result, BlockMatrix)
    expected = BlockMatrix([[ms('A_00', n/2, m)], [ms('A_10', n/2, m)]])

    assert result == expected
    assert result.shape == A.shape
    assert result.blockshape == (2, 1)
    assert result.blocks[0, 0].shape == (n/2, m)

def test_symbol_multi_dim():
    result = block_symbol(A, {n: [n/2, n/2], m: [m/2, m/4, m/4]})
    assert isinstance(result, BlockMatrix)
    expected = BlockMatrix([
        [ms('A_00', n/2, m/2), ms('A_01', n/2, m/4), ms('A_02', n/2, m/4)],
        [ms('A_10', n/2, m/2), ms('A_11', n/2, m/4), ms('A_12', n/2, m/4)]])

    assert result == expected
    assert result.shape == A.shape

def test_block_symbol_simple():
    expr = y*A*B
    bexpr = block_cut(expr, {n: [n/2, n/2], m: [m/2, m/4, m/4]})
    assert isinstance(bexpr, MatrixExpr)


