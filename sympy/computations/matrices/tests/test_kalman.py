from kalman_comp import mathcomp, assumptions
from kalman import mu, Sigma, H, R, data
from sympy.computations.inplace import inplace_compile
from sympy.computations.matrices.blas import COPY
from sympy.computations.matrices.fortran import gen_fortran, build, dimensions


def test_fortran_code_generation():
    ic = inplace_compile(mathcomp, Copy=COPY)
    s = gen_fortran(ic, assumptions, input_order=(mu, Sigma, H, R, data))
    print s
    assert isinstance(s, str)
    f = build(ic, assumptions, 'kalman', input_order=(mu, Sigma, H, R, data))
    assert callable(f)
