from sympy.computations.inplace import ExprToken, inplace_compile
from sympy.computations.matrices.fortran import (nameof,
        unique_tokened_variables, build, getintent, sort_arguments)
from sympy.core import Symbol, S
from sympy.matrices.expressions import MatrixSymbol
from sympy.computations.matrices.blas import GEMM, AXPY

def test_nameof():
    assert nameof(ExprToken(1, 'hello')) == 1
    assert nameof(ExprToken(Symbol('x'), 'y')) == 'y'

def test_unique_tokened_variables():
    x,y,z = map(Symbol, 'xyz')
    vars = ExprToken(x, 'x'), ExprToken(y, 'y'), ExprToken(z, 'x')
    result = unique_tokened_variables(vars)
    assert len(result) == 2

def test_sort_arguments():
    n,m,k = map(Symbol, 'nmk')
    X = MatrixSymbol('X', n, m)
    Y = MatrixSymbol('Y', m, k)
    Z = MatrixSymbol('Z', n, k)
    aX, aY, aZ = args = ExprToken(X, 'X'), ExprToken(Y, 'Y'), ExprToken(Z, 'Z')
    order = Z, X
    print sort_arguments(args, order)
    assert tuple(sort_arguments(args, order)) == (aZ, aX, aY)

def test_gemm():
    alpha, beta = Symbol('alpha'), Symbol('Beta')
    n,m,k = map(Symbol, 'nmk')
    X = MatrixSymbol('X', n, m)
    Y = MatrixSymbol('Y', m, k)
    Z = MatrixSymbol('Z', n, k)

    c = GEMM(alpha, X, Y, beta, Z)
    ct = inplace_compile(c)

    assert getintent(ct, ct.outputs[0]) == 'inout'  # Z is inout
    assert getintent(ct, ct.inputs[0]) == 'in'      # alpha is just in

    fn = build(ct, True, 'my_dgemm', input_order=(alpha, X, Y, beta, Z))

    # Check order of inputs is preserved
    assert "alpha,x,y,beta,z" in fn.__doc__.lower()

    try:
        n, m, k = 50, 40, 30
        alpha, beta = 2.5, 3.2
        import numpy as np
        X = np.array(np.random.rand(n, m), order='F')
        Y = np.array(np.random.rand(m, k), order='F')
        Z = np.array(np.random.rand(n, k), order='F')
        numpy_result = alpha*np.dot(X, Y) + beta*Z
        fn(alpha, X, Y, beta, Z)
        sympy_result = Z
        assert np.linalg.norm(numpy_result - sympy_result) < .01
    except ImportError:
        pass

def test_axpy():
    alpha = Symbol('alpha')
    n,m = map(Symbol, 'nm')
    W = MatrixSymbol('W', n, m)
    X = MatrixSymbol('X', n, m)
    c = AXPY(alpha, W, X)
    ct = inplace_compile(c)
    fn = build(ct, True, 'faxpy', input_order=(alpha, W, X))
    try:
        n, m = 50, 40
        alpha = 2.5
        import numpy as np
        W = np.array(np.random.rand(n, m), order='F')
        X = np.array(np.random.rand(n, m), order='F')
        numpy_result = alpha*W+X
        fn(alpha, W, X)
        sympy_result = X
        assert np.linalg.norm(numpy_result - sympy_result) < .01
    except ImportError:
        pass

def test_gemm_axpy():
    alpha, beta = Symbol('alpha'), Symbol('Beta')
    n,m,k = map(Symbol, 'nmk')
    W = MatrixSymbol('W', n, m)
    X = MatrixSymbol('X', n, m)
    Y = MatrixSymbol('Y', m, k)
    Z = MatrixSymbol('Z', n, k)

    c = AXPY(S.One, W, X) + GEMM(alpha, W+X, Y, beta, Z)
    ct = inplace_compile(c)

    fn = build(ct, True, 'gemmaxpy', input_order=(alpha, W, X, Y, beta, Z))
    return fn

    # Check order of inputs is preserved
    assert "alpha,w,x,y,beta,z" in fn.__doc__.lower()

    try:
        n, m, k = 50, 40, 30
        alpha, beta = 2.5, 3.2
        import numpy as np
        W = np.array(np.random.rand(n, m), order='F')
        X = np.array(np.random.rand(n, m), order='F')
        Y = np.array(np.random.rand(m, k), order='F')
        Z = np.array(np.random.rand(n, k), order='F')
        numpy_result = alpha*np.dot(W+X, Y) + beta*Z
        fn(alpha, W, X, Y, beta, Z)
        sympy_result = Z
        print numpy_result
        print sympy_result
        assert np.linalg.norm(numpy_result - sympy_result) < .01
    except ImportError:
        pass
