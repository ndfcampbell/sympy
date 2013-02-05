from kalman import newmu, newSigma, assumptions, mu, Sigma, R, H, data

from sympy.computations.matrices.compile import make_rule, patterns
from sympy.computations.core import Identity
from sympy.assumptions import assuming

ident = Identity(newmu, newSigma)
rule = make_rule(patterns)
with assuming(*assumptions):
    mathcomp = next(rule(ident))

if __name__ == '__main__':
    mathcomp.show()
    assert set(mathcomp.inputs) == set((mu, Sigma, H, R, data))
    assert set(mathcomp.outputs).issuperset(set((newmu, newSigma)))

