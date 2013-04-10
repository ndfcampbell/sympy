from sympy.computations.matrices.blas import GEMM, SYMM, AXPY
from sympy.computations.matrices.lapack import GESV, POSV, IPIV, LASWP
from sympy.computations.matrices.shared import (alpha, beta, n, m, k, A, B, C,
        x, a, b, X, Y, Z)
from sympy import Q, S, ask, Expr, Symbol, Dummy
from sympy.matrices.expressions import (MatrixExpr, PermutationMatrix,
        MatrixSymbol, ZeroMatrix, MatrixSlice)
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

def good_computation(c):
    """ Our definition of an acceptable computation

    Must:
        contain only symbols and matrix symbols as inputs
    """
    return all(isinstance(inp, (Symbol, MatrixSymbol)) or
               isinstance(inp, MatrixSlice) and
                    isinstance(inp.parent, MatrixSymbol)
               for inp in c.variable_inputs)

# pattern is (source expression, target expression, wilds, condition)
blas_patterns = [
    (alpha*A*B + beta*C, SYMM(*SYMM._inputs), SYMM._inputs, SYMM.condition),
    (alpha*A*B + C, SYMM(alpha, A, B, S.One, C), (alpha, A, B, C), SYMM.condition),
    (A*B + beta*C, SYMM(S.One, A, B, beta, C), (A, B, beta, C), SYMM.condition),
    (A*B + C, SYMM(S.One, A, B, S.One, C), (A, B, C), SYMM.condition),
    (alpha*A*B, SYMM(alpha, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (alpha, A, B), SYMM.condition),
    (A*B, SYMM(S.One, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (A, B), SYMM.condition),

    (alpha*A*B + beta*C, GEMM(*GEMM._inputs), GEMM._inputs, True),
    (alpha*A*B + C, GEMM(alpha, A, B, S.One, C), (alpha, A, B, C), True),
    (A*B + beta*C, GEMM(S.One, A, B, beta, C), (A, B, beta, C), True),
    (A*B + C, GEMM(S.One, A, B, S.One, C), (A, B, C), True),
    (alpha*A*B, GEMM(alpha, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (alpha, A, B), True),
    (A*B, GEMM(S.One, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (A, B), True),

    (alpha*X + Y, AXPY(*AXPY._inputs), AXPY._inputs, AXPY.condition),
    (X + Y, AXPY(S.One, X, Y), (X, Y), True)
]
lapack_patterns = [
    (Z.I*X, POSV(Z, X), (Z, X), Q.symmetric(Z) & Q.positive_definite(Z)),
    (Z.I*X, GESV(Z, X) + LASWP(PermutationMatrix(IPIV(Z.I*X))*Z.I*X, IPIV(Z.I*X)), (Z, X), True),

]

multi_out_patterns = [
]


patterns = lapack_patterns + blas_patterns

def makecond(wilds, assume):
    """ Trasform a Sympy Predicate object into a function

    inputs:
        wilds - variables in the predicate   like [x]
        assume - a SymPy predicate like Q.positive(x)

    outputs:
        a python function
        in the example above it's equivalent to lambda x: x > 0
    """
    return lambda *args: (typecheck(wilds, args) and
            (assume==True or ask(assume.xreplace(dict(zip(wilds, args))))))


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


compile = sfilter(good_computation, exhaust(multiplex(multioutrule, inrule)))

def compile(inputs, outputs):
    """
    Transform SymPy input/output expressions into sequence of Computations
    """
    incomp = Identity(*outputs)
    outcomps = exhaust(multiplex(multioutrule, inrule))(incomp)
    return (c for c in outcomps if set(c.variable_inputs) == set(inputs))
