from sympy.matrices.expressions.blas import GEMM, TRSV
from sympy.unify.unify_sympy import unify, patternify
from sympy.unify.rewrite import rewriterule
from sympy.rules.branch import multiplex
from sympy import MatrixSymbol

def rr_from_blas(cls):
    pattern_in = patternify(cls._outputs[0], *cls._inputs)
    pattern_out = cls(*cls._inputs)
    return rewriterule(pattern_in, pattern_out)

blas_rules = map(rr_from_blas, (GEMM, TRSV))
blas_rule = multiplex(*blas_rules)
