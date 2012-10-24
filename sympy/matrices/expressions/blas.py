from sympy.computations import Computation
from sympy import Basic, Tuple

class MM(Computation):
    def __new__(cls, alpha, A, B, beta, C):
        inputs  = (alpha, A, B, beta, C)
        outputs = (alpha * A * B + beta * C,)
        varouts = (C,)
        return Computation.__new__(cls, inputs, outputs, inputs, varouts)

class MV(MM):
    pass

class SM(Computation):
    def __new__(cls, alpha, A, B):
        inputs  = (alpha, A, B)
        outputs = (alpha * A.I * B,)
        varouts = (B,)
        return Computation.__new__(cls, inputs, outputs, inputs, varouts)

class SV(Computation):
    def __new__(cls, A, b):
        inputs  = (A, b)
        outputs = (A.I*b,)
        varouts = (b,)
        return Computation.__new__(cls, inputs, outputs, inputs, varouts)

class Alloc(Computation):
    def __new__(cls, x):
        inputs  = ()
        outputs = (x,)
        varouts = (x,)
        return Computation.__new__(cls, inputs, outputs, inputs, varouts)

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
        cp = _copy(x)
        varouts = (cp,)
        return Computation.__new__(cls, inputs, outputs, inputs, varouts)

class LUDecomposition(Computation):
    def __new__(cls, A):
        inputs  = (A,)
        outputs = (Lof(A), Uof(A))
        varouts = (A, A)
