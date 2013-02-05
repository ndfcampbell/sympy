from kalman import newmu, newSigma, assumptions, mu, Sigma, R, H, data

from sympy.computations.matrices.compile import compile
from sympy.computations.core import Identity
from sympy.assumptions import assuming

ident = Identity(newmu, newSigma)
with assuming(*assumptions):
    mathcomp = next(compile(ident))

if __name__ == '__main__':
    mathcomp.show()
    assert set(mathcomp.inputs) == set((mu, Sigma, H, R, data))
    assert set(mathcomp.outputs).issuperset(set((newmu, newSigma)))

