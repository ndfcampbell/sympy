from mvnrv import SingleMultivariatePSpace
from crv_examples import Normal as SingleNormal
from sympy import sympify

__all__ = ['MultivariateNormal']
#__all__ = ['Normal', 'MultivariateNormal']

def MultivariateNormal(mean, covariance, symbol=None):
    return SingleMultivariatePSpace(mean, covariance, symbol).value

def Normal(mean, spread, symbol=None):
    if sympify(mean).is_Matrix:
        return MultivariateNormal(mean, spread, symbol)
    else:
        return SingleNormal(mean, spread, symbol)

