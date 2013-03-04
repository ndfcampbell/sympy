from sympy.computations.matrices.blas import GEMM, SYMM, AXPY
from sympy.computations.matrices.lapack import GESV, POSV, IPIV, LASWP
from sympy.computations.matrices.shared import (alpha, beta, n, m, k, A, B, C,
        x, a, b, X, Y, Z)
from sympy import Q, S, ask, Expr, Symbol, Dummy
from sympy.matrices.expressions import (MatrixExpr, PermutationMatrix,
        MatrixSymbol, ZeroMatrix, MatrixSlice)
from sympy.computations.compile import input_crunch, multi_output_rule
from sympy.unify import rewriterule
from sympy.unify.usympy import types
from sympy.unify.rewrites import rewriterules
from sympy.rules.branch import multiplex, exhaust, debug, sfilter, condition
from sympy.rules.util import count
from functools import partial

basetypes = (Expr, MatrixExpr)
def basetype(var):
    for bt in basetypes:
        if isinstance(var, bt):
            return bt

def typecheck(wilds):
    def check(*variables):
        return all(basetype(v) == basetype(w)
                   for v, w in zip(variables, wilds))
    return check

def good_computation(c):
    """ Our definition of an acceptable computation

    Must:
        contain only symbols and matrix symbols as inputs
    """
    return all(isinstance(inp, (Symbol, MatrixSymbol)) or
               isinstance(inp, MatrixSlice) and
                    isinstance(inp.parent, MatrixSymbol)
               for inp in c.inputs)

# pattern is (source expression, target expression, wilds, condition)
blas_patterns = [
    (alpha*A*B + beta*C, SYMM(*SYMM._inputs), SYMM._inputs, SYMM.condition),
    (alpha*A*B + C, SYMM(alpha, A, B, S.One, C), (alpha, A, B, C), SYMM.condition),
    (A*B + beta*C, SYMM(S.One, A, B, beta, C), (A, B, beta, C), SYMM.condition),
    (A*B + C, SYMM(S.One, A, B, S.One, C), (A, B, C), SYMM.condition),
    (alpha*A*B, SYMM(alpha, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (alpha, A, B), SYMM.condition),
    (A*B, SYMM(S.One, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (A, B), SYMM.condition),

    (alpha*A*B + beta*C, GEMM(*GEMM._inputs), GEMM._inputs, GEMM.condition),
    (alpha*A*B + C, GEMM(alpha, A, B, S.One, C), (alpha, A, B, C), True),
    (A*B + beta*C, GEMM(S.One, A, B, beta, C), (A, B, beta, C), True),
    (A*B + C, GEMM(S.One, A, B, S.One, C), (A, B, C), True),
    (alpha*A*B, GEMM(alpha, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (alpha, A, B), True),
    (A*B, GEMM(S.One, A, B, S.Zero, ZeroMatrix(A.rows, B.cols)), (A, B), True),

    (alpha*X + Y, AXPY(*AXPY._inputs), AXPY._inputs, AXPY.condition),
    (X + Y, AXPY(S.One, X, Y), (X, Y), True)
]
lapack_patterns = [
    (POSV._outputs[0], POSV(*POSV._inputs), POSV._inputs, POSV.condition),
    (Z.I*X, GESV(Z, X) + LASWP(PermutationMatrix(IPIV(Z.I*X))*Z.I*X, IPIV(Z.I*X)), (Z, X), True),

]

multi_out_patterns = [
]


patterns = lapack_patterns + blas_patterns

def makecond(wilds, assume):
    return lambda *args: (typecheck(wilds)(*args) and
            (assume==True or ask(assume.xreplace(dict(zip(wilds, args))))))


replace = {MatrixSymbol: MatrixExpr, Symbol: Expr, Dummy: Expr,
           MatrixSlice: MatrixExpr}
types = partial(types, replace=replace)
def makerule(pattern):
    s, t, wilds, assume = pattern
    cond = makecond(wilds, assume)
    typecounts = count(types(s))
    typecond = lambda e: all(count(types(e)).get(k, 0) >= v
                                for k, v in typecounts.items())
    return condition(typecond, rewriterule(s, t, wilds, cond))

rules = map(makerule, patterns)

inrule = input_crunch(multiplex(*rules))

multioutrules = [multi_output_rule(sources, target, *wilds)
            for sources, target, wilds, condition in multi_out_patterns]
multioutrule = multiplex(*multioutrules)


from sympy.rules.branch import onaction
def makepdf(brl, expr, result):
    result.show()
pdfdebug = partial(onaction, fn=makepdf)


compile = sfilter(good_computation, exhaust(multiplex(multioutrule, inrule)))
