from sympy.computations.matrices.blas import GEMM, SYMM, AXPY, SYRK
from sympy.computations.matrices.lapack import GESV, POSV, IPIV, LASWP
from sympy.computations.matrices.fftw import FFTW
from sympy.computations.matrices.blocks import JoinBlocks, Slice
from sympy.computations.matrices.shared import (alpha, beta, n, m, k, A, B, C,
        x, a, b, X, Y, Z)
from sympy import Q, S, ask, Expr, Symbol, Dummy, Integer
from sympy.logic.boolalg import Boolean
from sympy.matrices.expressions import (MatrixExpr, PermutationMatrix,
        MatrixSymbol, ZeroMatrix, MatrixSlice, BlockMatrix)
from sympy.matrices.expressions.fourier import DFT
from sympy.computations.compile import input_crunch, multi_output_rule
from sympy.computations.core import Identity
from sympy.unify import rewriterule
from sympy.unify.usympy import subtypes as types
from sympy.unify.rewrites import rewriterules
from sympy.strategies.branch import multiplex, exhaust, debug, sfilter, condition
from sympy.strategies.util import count
from functools import partial

basetypes = (Expr, MatrixExpr)
def basetype(var):
    """ Return main super-class for SymPy object

    Returns either Expr       for scalar expressions
                of MatrixExpr for matrix expressions
    """
    for bt in basetypes:
        if isinstance(var, bt):
            return bt

def typecheck(wilds, variables):
    return all(basetype(v) == basetype(w)
               for v, w in zip(variables, wilds))


# pattern is (source expression, target expression, wilds, condition)
blas_patterns = [
    (A*A.T, SYRK(1.0, A, 0.0, ZeroMatrix(A.rows, A.rows)), (A,), True),
    (A.T*A, SYRK(1.0, A.T, 0.0, ZeroMatrix(A.cols, A.cols)), (A,), True),
    (alpha*A*B + beta*C, SYMM(*SYMM._inputs), SYMM._inputs, SYMM.condition),
    (alpha*A*B + C, SYMM(alpha, A, B, 1.0, C), (alpha, A, B, C), SYMM.condition),
    (A*B + beta*C, SYMM(1.0, A, B, beta, C), (A, B, beta, C), SYMM.condition),
    (A*B + C, SYMM(1.0, A, B, 1.0, C), (A, B, C), SYMM.condition),
    (alpha*A*B, SYMM(alpha, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), (alpha, A, B), SYMM.condition),
    (A*B, SYMM(1.0, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), (A, B), SYMM.condition),

    (alpha*A*B + beta*C, GEMM(*GEMM._inputs), GEMM._inputs, True),
    (alpha*A*B + C, GEMM(alpha, A, B, 1.0, C), (alpha, A, B, C), True),
    (A*B + beta*C, GEMM(1.0, A, B, beta, C), (A, B, beta, C), True),
    (A*B + C, GEMM(1.0, A, B, 1.0, C), (A, B, C), True),
    (alpha*A*B, GEMM(alpha, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), (alpha, A, B), True),
    (A*B, GEMM(1.0, A, B, 0.0, ZeroMatrix(A.rows, B.cols)), (A, B), True),

    (alpha*X + Y, AXPY(*AXPY._inputs), AXPY._inputs, AXPY.condition),
    (X + Y, AXPY(1.0, X, Y), (X, Y), True)
]
lapack_patterns = [
    (Z.I*X, POSV(Z, X), (Z, X), Q.symmetric(Z) & Q.positive_definite(Z)),
    (Z.I*X, GESV(Z, X) + LASWP(PermutationMatrix(IPIV(Z.I*X))*Z.I*X, IPIV(Z.I*X)), (Z, X), True),

]

multi_out_patterns = [
]

ints = start1, stop1, step1, start2, stop2, step2 = map(Dummy,
        '_start1 _stop1 _step1 _start2 _stop2 _step2'.split())
other_patterns = [
    (DFT(n) * x, FFTW(x), (x, n), True),
    (X, JoinBlocks(X), (X,), lambda x: isinstance(x, BlockMatrix)),
    (X[start1:stop1:step1, start2:stop2:step2],
        Slice(X[start1:stop1:step1, start2:stop2:step2]), (X,) + tuple(ints), True),
]

patterns = lapack_patterns + blas_patterns + other_patterns

def makecond(wilds, assume):
    """ Trasform a Sympy Predicate object into a function

    inputs:
        wilds - variables in the predicate   like [x]
        assume - a SymPy predicate like Q.positive(x)

    outputs:
        a python function
        in the example above it's equivalent to lambda x: x > 0
    """
    def cond(*args):
        if not typecheck(wilds, args):
            return False
        if assume == True:
            return True
        if isinstance(assume, Boolean):
            return ask(assume.xreplace(dict(zip(wilds, args))))
        if callable(assume):
            return assume(*args)
        raise TypeError()

    return cond


replace = {MatrixSymbol: MatrixExpr, Symbol: Expr, Dummy: Expr,
           MatrixSlice: MatrixExpr}
types = partial(types, replace=replace)
def makerule((source, target, wilds, assume)):
    """ Transform a pattern to a transformation rule

    Counts frequency of types before applying rule for efficiency
    """
    cond = makecond(wilds, assume)
    typecounts = count(types(source))
    typecond = lambda e: all(count(types(e)).get(k, 0) >= v
                                for k, v in typecounts.items())
    return condition(typecond, rewriterule(source, target, wilds, cond))

rules = map(makerule, patterns)

inrule = input_crunch(multiplex(*rules))

multioutrules = [multi_output_rule(sources, target, *wilds)
            for sources, target, wilds, condition in multi_out_patterns]
multioutrule = multiplex(*multioutrules)


from sympy.strategies.branch import onaction
def makepdf(brl, expr, result):
    result.show()
pdfdebug = partial(onaction, fn=makepdf)


def compile(inputs, outputs):
    """
    Transform SymPy input/output expressions into sequence of Computations
    """
    incomp = Identity(*outputs)
    outcomps = exhaust((multiplex(multioutrule, inrule)))(incomp)
    return (c for c in outcomps if set(c.variable_inputs) == set(inputs))
