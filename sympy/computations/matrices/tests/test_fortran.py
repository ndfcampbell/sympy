from sympy.computations.inplace import ExprToken, inplace_compile
from sympy.computations.matrices.fortran import (nameof,
        unique_tokened_variables, build, getintent)
from sympy.core import Symbol
from sympy.matrices.expressions import MatrixSymbol
from sympy.computations.matrices.blas import GEMM

def test_nameof():
    assert nameof(ExprToken(1, 'hello')) == 1
    assert nameof(ExprToken(Symbol('x'), 'y')) == 'y'

def test_unique_tokened_variables():
    x,y,z = map(Symbol, 'xyz')
    vars = ExprToken(x, 'x'), ExprToken(y, 'y'), ExprToken(z, 'x')
    result = unique_tokened_variables(vars)
    assert len(result) == 2

def test_gemm():
    alpha, beta = Symbol('alpha'), Symbol('Beta')
    n,m,k = map(Symbol, 'nmk')
    X = MatrixSymbol('X', n, m)
    Y = MatrixSymbol('Y', m, k)
    Z = MatrixSymbol('Z', n, k)

    c = GEMM(alpha, X, Y, beta, Z)
    ct = inplace_compile(c)

    assert getintent(ct, ct.outputs[0]) == 'inout'
    assert getintent(ct, ct.inputs[0]) == 'in'

    fn = build(ct, True, 'my_dgemm')
