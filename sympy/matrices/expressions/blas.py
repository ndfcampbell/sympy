from sympy.computations import Computation
from sympy import Basic, Tuple

def update(a, b):
    return a.copy().update(b)

alpha = Symbol('alpha')
beta  = Symbol('beta')
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('A', m, k)
C = MatrixSymbol('C', n, k)
S = MatrixSymbol('S', n, n)
x = MatrixSymbol('x', n, 1)
a = MatrixSymbol('a', m, 1)
b = MatrixSymbol('b', k, 1)

mm = {'inputs':  (alpha, A, B, beta, C),
      'outputs': (alpha*A*B + beta*C,),
      'view_map': {0:4},
      'condition': True}

gemm = mm
symm = update(mm, {'condition': Q.symmetric(A) | Q.symmetric(B)})
trmm = update(mm, {'condition': Q.triangular(A) | Q.triangular(B)})

sm = {'inputs'   = (alpha, A, B),
      'outputs'  = (alpha * A.I * B,),
      'view_map' = {0: 2},
      'condition'= True}
trsm = update(sm, {'condition': Q.triangular(A)})

mv = {'inputs':  (alpha, A, a, beta, b),
      'outputs': (alpha*A*a + beta*b,),
      'view_map': {0:4},
      'condition': True}
gemv = mm
symv = update(mm, {'condition': Q.symmetric(A) | Q.symmetric(B)})
trmv = update(mm, {'condition': Q.triangular(A) | Q.triangular(B)})

sv = {'inputs'   = (A, x),
      'outputs'  = (A.I*x,),
      'view_map' = {0: 1},
      'condition'= True}
trsv = update(sv, {'condition': Q.triangular(A)})

lu = {'inputs': (S,),
      'outputs': (Lof(S), Uof(S)),
      'view_map': {0:0, 1:0},
      'condition': Q.square(S)}
cholesky = update(lu, {'condition': Q.symmetric(S) & Q.positive_definite(S)})

class BLASopPure(Computation):
    def __new__(cls, d):
        inputs = d['inputs']
        outputs = d['outputs']
        return Computation.__new__(cls, inputs, outputs, {})

class BLASop(Computation):
    def __new__(cls, d):
        inputs = d['inputs']
        outputs = d['outputs']
        view_map = d['view_map']
        return Computation.__new__(cls, inputs, outputs, view_map)

class Alloc(Computation):
    def __new__(cls, x):
        inputs  = ()
        outputs = (x,)
        return Computation.__new__(cls, inputs, outputs, {})

class _copy(Basic):
    _id = 0
    def __new__(cls, x, id=None):
        if id is None:
            id = _copy._id
        _copy._id = max(_copy._id, id + 1)
        return Basic.__new__(cls, x, id)

class Copy(Computation):
    def __new__(cls, x):
        inputs  = (x,)
        outputs = (x,)
        return Computation.__new__(cls, inputs, outputs, {})
