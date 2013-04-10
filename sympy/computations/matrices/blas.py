from sympy.computations.matrices.core import MatrixCall, remove_numbers
from sympy.computations.core import unique
from sympy.computations.inplace import Copy
from sympy.computations.matrices.shared import (detranspose, trans, LD,
        left_or_right, diag)
from sympy.computations.matrices.shared import (alpha, beta, n, m, k, A, B, C,
        x, a, b, X, Y)
from sympy import Q, S
from sympy.utilities.iterables import dict_merge as merge
from sympy.matrices.expressions import ZeroMatrix

class BLAS(MatrixCall):
    """ Basic Linear Algebra Subroutine - Dense Matrix computation """
    flags = ["-lblas"]

class MM(BLAS):
    """ Matrix Multiply """
    _inputs   = (alpha, A, B, beta, C)
    _outputs  = (alpha*A*B + beta*C,)
    inplace   = {0: 4}
    condition = True

    @property
    def inputs(self):
        alpha, A, B, beta, C, typecode = self.args
        if isinstance(C, ZeroMatrix):    # special case this
            C = ZeroMatrix(A.rows, B.cols)
        # Sometimes we use C only as an output. It should be detransposed
        A = detranspose(A)
        B = detranspose(B)
        C = detranspose(C)
        return alpha, A, B, beta, C

    @property
    def outputs(self):
        alpha, A, B, beta, C, typecode = self.args
        if isinstance(C, ZeroMatrix):    # special case this
            C = ZeroMatrix(A.rows, B.cols)
        return (alpha*A*B + beta*C,)

    @classmethod
    def codemap(cls, inputs, names, typecode, assumptions=True):
        varnames = 'alpha A B beta C'.split()
        alpha, A, B, beta, C = inputs

        namemap  = dict(zip(varnames, names))
        other = {'TRANSA': trans(A), 'TRANSB': trans(B),
                 'LDA': LD(A), 'LDB': LD(B), 'LDC': LD(C),
                 'M':str(C.shape[0]), 'K':str(B.shape[0]), 'N':str(C.shape[1]),
                 'fn': cls.fnname(typecode),
                 'SIDE': left_or_right(A, B, Q.symmetric, assumptions),
                 'DIAG': diag(A, assumptions),
                 'UPLO': 'U'} # TODO: symmetric matrices might be stored low
        return merge(namemap, other)


class GEMM(MM):
    """ General Matrix Multiply """
    fortran_template = ("call %(fn)s('%(TRANSA)s', '%(TRANSB)s', "
                        "%(M)s, %(N)s, %(K)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, "
                        "%(B)s, %(LDB)s, %(beta)s, %(C)s, %(LDC)s)")

class SYMM(MM):
    """ Symmetric Matrix Multiply """
    condition = Q.symmetric(A) | Q.symmetric(B)
    fortran_template = ("call %(fn)s('%(SIDE)s', '%(UPLO)s', %(M)s, %(N)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, %(B)s, %(LDB)s, "
                        "%(beta)s, %(C)s, %(LDC)s)")

class AXPY(BLAS):
    """ Matrix Matrix Addition `alpha X + Y` """
    _inputs   = (alpha, X, Y)
    _outputs  = (alpha*X + Y,)
    inplace   = {0: 2}
    condition = True

    fortran_template = ("call %(fn)s(%(N)s, %(alpha)s, %(A)s, "
                        "%(INCX)d, %(B)s, %(INCY)d)")

    @classmethod
    def codemap(cls, inputs, names, typecode, assumptions=True):
        varnames = 'alpha A B'.split()
        alpha, A, B = inputs

        namemap  = dict(zip(varnames, names))
        other = {'N': A.rows*A.cols,
                 'fn': cls.fnname(typecode),
                 'INCX': 1,
                 'INCY': 1}
        return merge(namemap, other)

class COPY(BLAS, Copy):
    """ Array to array copy """
    _inputs   = (X,)
    _outputs  = (X,)

    fortran_template = "call %(fn)s(%(N)s, %(X)s, %(INCX)s, %(Y)s, %(INCY)s)"

    @classmethod
    def codemap(cls, inputs, names, typecode, assumptions=True):
        varnames = 'X Y'.split()
        X, = inputs

        namemap  = dict(zip(varnames, names))
        other = {'N': X.rows*X.cols,
                 'fn': cls.fnname(typecode),
                 'INCX': 1,
                 'INCY': 1}
        return merge(namemap, other)

    @staticmethod
    def arguments(inputs, outputs):
        return inputs + outputs
