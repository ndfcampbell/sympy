from sympy.matrices.expressions.blas import GEMM, TRSV, SYMM
from sympy.matrices.expressions.lapack import GESV, POSV
from sympy.unify.usympy import unify, patternify
from sympy.unify.rewrite import rewriterule
from sympy.rules.branch import multiplex, condition
from sympy import MatrixSymbol, MatrixExpr
from sympy import ask, Q

def rr_from_blas(cls, assumptions=True):
    pattern_in = patternify(cls._outputs[0], *cls._inputs)
    pattern_out = cls(*cls._inputs)
    rr = rewriterule(pattern_in, pattern_out)
    def blas_brl(expr):
        for blas in rr(expr):
            if cls.valid(blas.raw_inputs, assumptions):
                yield blas
    return blas_brl


classes = (SYMM, GEMM, TRSV, POSV, GESV)
def build_rule(assumptions, classes=classes):
    return multiplex(*[rr_from_blas(cls, assumptions) for cls in classes])

# blas_rules = map(rr_from_blas, (GEMM, TRSV))
# blas_rule = multiplex(*blas_rules)

from sympy.rules.branch.strat_pure import notempty
from sympy.core.compatibility import product
from sympy.computations import Computation

new = lambda ne, *cs: ne._composite(cs)
is_leaf = lambda x: not isinstance(x, Computation)
children = lambda x: x.inputs

def top_down(brule):
    """ Apply a rule down a tree running it on the top nodes first """
    ident = "_identity"
    def top_down_rl(expr):
        # print "In:  ", expr
        anything = False
        for comp in brule(expr):
            anything = True
            for comps in product(*map(top_down_rl, comp.inputs)):
                x = comp._composite(filter(lambda x: not x is ident, comps) +
                                      (comp,))
                # print "Out: ", x
                yield x
        if not anything:
            yield ident
    return top_down_rl
