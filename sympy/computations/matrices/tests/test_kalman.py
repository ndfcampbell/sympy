from kalman_comp import mathcomp, assumptions
from kalman import mu, Sigma, H, R, data
from sympy.computations.inplace import inplace_compile
from sympy.computations.matrices.blas import COPY
from sympy.computations.matrices.fortran import gen_fortran, build, dimensions
from sympy.assumptions import assuming


def test_fortran_code_generation():
    ic = inplace_compile(mathcomp, Copy=COPY)
    mathcomp.writepdf('kalman.math')
    ic.writepdf('kalman')
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
