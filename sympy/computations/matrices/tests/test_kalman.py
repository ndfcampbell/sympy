from kalman_comp import mathcomp, assumptions
from kalman import mu, Sigma, H, R, data, inputs, outputs
from sympy.computations.inplace import inplace_compile
from sympy.computations.matrices.blas import COPY
from sympy.computations.matrices.fortran import gen_fortran, build, dimensions
from sympy.computations.matrices.fortran2 import generate_fortran
from sympy.assumptions import assuming
from sympy.computations.dot import writepdf
from sympy import Symbol


def test_fortran_code_generation():

    writepdf(mathcomp, 'kalman.math')
    writepdf(ic, 'kalman')
    with assuming(*assumptions):
        s = gen_fortran(ic, input_order=(mu, Sigma, H, R, data))
        with open('kalman.f90', 'w') as f:
            f.write(s)
        assert isinstance(s, str)
        f = build(ic, 'kalman', input_order=(mu, Sigma, H, R, data))

def test_kalman_run():
    ic = inplace_compile(mathcomp, Copy=COPY)
    with assuming(*assumptions):
        f = build(ic, 'kalman', input_order=(mu, Sigma, H, R, data))
    try:
        n, m, k = 50, 40, 30
        alpha, beta = 2.5, 3.2
        import numpy as np
        nSigma = np.array(np.random.rand(n, n), order='F')
        nmu = np.array(np.random.rand(n), order='F')
        nH = np.array(np.random.rand(k, n), order='F')
        nR = np.array(np.random.rand(k, k), order='F')
        ndata = np.array(np.random.rand(k), order='F')
        output = f(nmu, nSigma, nH, nR, ndata)
        print output
    except ImportError:
        pass

ic = inplace_compile(mathcomp, Copy=COPY)
types = {v: 'real*8' for v in tuple(mathcomp.variables)}
types[Symbol('INFO')] = 'integer'
with assuming(*assumptions):
    s = generate_fortran(ic, inputs, outputs, types, 'kalman')

def test_declarations():
    assert isinstance(s, str)
    assert \
"""real*8, intent(in) :: mu(n)
real*8, intent(in) :: Sigma(n, n)
real*8, intent(in) :: H(k, n)
real*8, intent(in) :: R(k, k)
real*8, intent(in) :: data(k)
real*8, intent(out) :: muvar_2(n)
real*8, intent(out) :: Sigmavar_2(n, n)
""" in s

    assert "integer :: INFO" in s
