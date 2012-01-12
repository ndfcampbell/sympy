from sympy.stats import (Normal, Exponential, P, E, Where, Density, Var, Covar,
        Skewness, Gamma, Pareto, Beta, Uniform, Given, pspace, CDF,
        ContinuousRV, Sample)
from sympy import (Symbol, exp, S, pi, simplify, Interval, erf, Eq, symbols,
        sqrt, And, gamma, beta, Piecewise)
from sympy.utilities.pytest import raises
from sympy.stats.mvnrv_types import MultivariateNormal as Normal

def test_Kalman():
    mu = MatrixSymbol('mu', n, 1) # n by 1 mean vector
    Sigma = MatrixSymbol('Sigma', n, n) # n by n covariance matrix
    X = Normal(mu, Sigma, 'X') # a multivariate normal random variable

    assert Density(X) == (mu, Sigma)

    H = MatrixSymbol('H', k, n) # A linear operator
    assert Density(H*X) == (H*mu, H*Sigma*H.T)

    # Lets make some measurement noise
    zerok = ZeroMatrix(k, 1) # mean zero
    R = MatrixSymbol('R', k, k) # symbolic covariance matrix
    noise = Normal(zerok, R, 'eta')

    assert Density(H*X + noise)

    assert block_collapse(Density(H*X + noise)) == (H*mu, R + H*Sigma*H.T)

    # Now lets imagine that we observe some value of HX+noise,
    # what does that tell us about X? How does our prior distribution change?
    data = MatrixSymbol('data', k, 1)
    assert block_collapse(Density(X,  Eq(H*X+noise, data))) ==

