from sympy.matrices.expressions import MatrixSymbol, Transpose
from sympy import Symbol

# Pattern variables
alpha = Symbol('alpha')
beta  = Symbol('beta')
n,m,k = map(Symbol, 'nmk')
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, k)
C = MatrixSymbol('C', n, k)
S = MatrixSymbol('S', n, n)
x = MatrixSymbol('x', n, 1)
a = MatrixSymbol('a', m, 1)
b = MatrixSymbol('b', k, 1)

def detranspose(A):
    """ Unpack a transposed matrix """
    if isinstance(A, Transpose):
        return A.arg
    else:
        return A
