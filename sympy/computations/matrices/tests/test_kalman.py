from kalman_comp import mathcomp, assumptions
from sympy.computations.inplace import inplace_compile
from sympy.computations.matrices.blas import COPY
from sympy.computations.matrices.fortran import gen_fortran, build, dimensions


def test_fortran_code_generation():
    ic = inplace_compile(mathcomp, Copy=COPY)
    s = gen_fortran(ic, assumptions)
    print s
    assert isinstance(s, str)
    f = build(ic, assumptions, 'kalman')
    assert callable(f)


