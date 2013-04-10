from kalman import newmu, newSigma, assumptions, mu, Sigma, R, H, data

from sympy.computations.matrices.compile import compile
from sympy.computations.core import Identity
from sympy.assumptions import assuming

ident = Identity(newmu, newSigma)
with assuming(*assumptions):
    mathcomp = next(compile([mu, Sigma, R, H, data], [newmu, newSigma]))

if __name__ == '__main__':
    from sympy.computations.dot import show
    show(mathcomp)
    assert set(mathcomp.variable_inputs) == set((mu, Sigma, H, R, data))
    assert set(mathcomp.outputs).issuperset(set((newmu, newSigma)))

