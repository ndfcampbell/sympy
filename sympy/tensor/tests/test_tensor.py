from sympy.tensor.tensor import *

i,j,k,l,m,n = indices('ijklmn')

def test_indexing():
    R = TensorSymbol('R', ((4,4), (4,4)))
    g = TensorSymbol('g', ((4,4), ()))

    assert isinstance(R, TensorSymbol)
    assert isinstance(R, Tensor)
    assert isinstance(R[i|j, k|l], IndexedTensor)
    assert R.rank == (2,2)
    assert g.rank == (2,0)
    assert R[i|j, k|l].rank == (2,2)
    R_lowered = (R[i|j, k|l]*g[k|m]*g[l|n])
    assert R_lowered.rank == (4,0)
    assert R_lowered.free_contravariants == FiniteSet(i,j,m,n)
    assert R_lowered.free_covariants     == EmptySet()

    assert R[i|j, j|k].rank == (1,1) # Contraction

def test_matrices():
    a,b,c,d= symbols('a,b,c,d')
    assert MatrixSymbol('X', a, b) == TensorSymbol('X', ((a,), (b,)))
    X = MatrixSymbol('X', a, b)
    Y = MatrixSymbol('Y', b, c)
    Z = MatrixSymbol('z', c, d)
    x = MatrixSymbol('x', d)

    def is_matrix(x):
        return x.rank == (1,1)
    def is_col_vector(x):
        return x.rank == (1,0)

    assert is_matrix(X)
    assert is_col_vector(x)
    assert is_matrix(X[i,j]*Y[j,k])
    assert is_col_vector(X[i,j]*Y[j,k]*Z[k,l]*x[l])
    assert (X[i,j]*Y[j,k]).rank






