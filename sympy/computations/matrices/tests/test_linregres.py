from sympy import MatrixSymbol, Q
n, m = 3, 2
X = MatrixSymbol('X', n, m)
y = MatrixSymbol('y', n, 1)
beta = (X.T*X).I * X.T*y

import numpy
nX = numpy.matrix(numpy.asarray([[2,3], [3,4], [4, 5]]))
ny = numpy.matrix(numpy.asarray([[1], [2], [3]]))
nbeta = (nX.T*nX).I*nX.T*ny

def test_fortran():
    from sympy.computations.matrices.fortran import fortran_function
    f = fortran_function([X, y], [beta], Q.fullrank(X))
    assert numpy.allclose(f(nX, ny)[1], numpy.asarray(nbeta).squeeze())
