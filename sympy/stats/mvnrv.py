"""
Multivariate Normal Random Variables Module

Multivariate normal random variables symbolically represent high-dimensional
vectors distributed according to a multivariate normal distribution.

The operations on these variable types are limited to application of linear
operators i.e. H*X where X is a MVNRV and H is a matrix.

This module was produced to highlight how the random variable framework could
be extended to new types. This class implements all logic behind the Kalman
Filter.

Warning: This code is not mature

See Also
========
sympy.stats.mvnrv_types
sympy.stats.rv
sympy.stats.crv
sympy.stats.frv
"""

from rv import (RandomDomain, SingleDomain, PSpace, ConditionalDomain,
        ProductPSpace, RandomSymbol)
from sympy import Interval, S, FiniteSet, Symbol, Tuple
from sympy.matrices import (BlockMatrix, BlockDiagMatrix, linear_factors,
        Transpose, MatrixSymbol, block_collapse, Inverse, Identity, ZeroMatrix)

oo = S.Infinity
Rn = Interval(-oo, oo)

def is_linear(expr, syms=None):
    return True

class MultivariateDomain(Domain):
    is_Multivariate = True

class SingleMultivariateDomain(MultivariateDomain, SingleDomain):
    def __new__(cls, symbol):
        assert symbol.is_Symbol
        symbols = FiniteSet(symbol)
        return Domain.__new__(cls, symbols, Rn)

    def as_boolean(self):
        return self.set.as_relational(self.symbol)

class ConditionalMultivariateDomain(MultivariateDomain, ConditionalDomain):
    pass

class MultivariatePSpace(PSpace):
    is_Multivariate = True

    @property
    def symbol(self):
        return self.density[0]
    @property
    def mean(self):
        return self.density[1]
    @property
    def covariance(self):
        return self.density[2]
    @property
    def values(self):
        return frozenset(RandomMatrixSymbol(self, sym)
                for sym in self.domain.symbols)

    def integrate(self, expr, rvs=None, **kwargs):
        raise NotImplementedError("Expectations not implemented in MVNRV case")
        #if rvs == None:
        #    rvs = self.values
        #else:
        #    rvs = frozenset(rvs)
        #
        #assert is_linear(expr, rvs)
        #return expr.subs{rv:mean_of_rv}

    def _expr_to_operator(self, expr):
        """
        Given an expression like HX+Y build a linear operator which creates
        this expression when applied onto the vector of random values of this
        object (assume in this example values = [X, Y])
        """
        expr = expr.subs(dict((rv, rv.symbol) for rv in self.values))

        d = linear_factors(expr, *self.symbols)

        # Construct row vector
        # Be careful because we may have blocks of blockmatrices in self.symbol
        def access_d(sym):
            if sym.is_Symbol:
                # Just return the coefficient (or zero if none)
                return d.get(sym, ZeroMatrix(expr.n, sym.n))
            if sym.is_BlockMatrix: # need to recursively return BlockMatrix
                return BlockMatrix([[access_d(s) for s in sym]])

        if self.symbol.is_BlockMatrix:
            # Operator is a block row vector
            operator = BlockMatrix([[access_d(sym) for sym in self.symbol]])
        else:
            operator = d[self.symbol]
        return operator

    def compute_density(self, expr, **kwargs):
        """
        Compute the mean and covariance matrices of a random expression
        """

        operator = self._expr_to_operator(expr)
        expr = expr.subs(dict((rv, rv.symbol) for rv in self.values))

        collapsed = block_collapse(operator * self.symbol)
        if collapsed.is_BlockMatrix:
            collapsed = collapsed[0]
        additive_terms = expr - collapsed
        mean = operator * self.mean + additive_terms
        covar = operator * self.covariance * Transpose(operator)

        return mean, covar

    def P(self, condition, **kwargs):
       raise NotImplementedError(
               "Operation not implemented for Multivariate case")

    def where(self, condition):
       raise NotImplementedError(
               "Operation not implemented for Multivariate case")

    def conditional_space(self, condition, **kwargs):
        if not condition.is_Equality:
            raise NotImplementedError("Only equality constraints supported")

        expr = condition.lhs - condition.rhs
        expr = expr.subs(dict((rv, rv.symbol) for rv in self.values))

        operator = self._expr_to_operator(expr)
        collapsed = block_collapse(operator * self.symbol)
        if collapsed.is_BlockMatrix:
            collapsed = collapsed[0]
        additive_terms = expr - collapsed

        # Kalman Syntax
        x = self.mean
        P = self.covariance
        H = operator
        z = additive_terms # data

        y = z - H*x
        S = H*P*Transpose(H)
        K = P*Transpose(H)*Inverse(S)
        xx = x + K*y
        PP = (Identity(P.n) - K*H)*P

        #ConditionalMultivariateDomain(self.domain, condition)
        domain = self.domain;
        density = (self.symbol, xx, PP)

        return MultivariatePSpace(domain, density)

class SingleMultivariatePSpace(MultivariatePSpace):
    _count = 0
    _name = 'X'
    def __new__(cls, mean, covariance, symbol=None):
        if not symbol:
            symbol = cls._name+str(cls._count)
            cls._count+=1
        if isinstance(symbol, str):
            symbol = MatrixSymbol(symbol, *mean.shape)
        assert symbol.is_Matrix
        if (symbol.shape!=mean.shape or mean.n != covariance.n
                or not covariance.is_square):
            raise ShapeError("symbol, mean, covariance have inconsistent shape")
        domain = SingleMultivariateDomain(symbol)
        #symbol = BlockMatrix([[symbol]])
        #mean = BlockMatrix([[mean]])
        #covariance = BlockMatrix([[covariance]])
        density = Tuple(symbol, mean, covariance)
        return MultivariatePSpace.__new__(cls, domain, density)

    @property
    def value(self):
        return tuple(self.values)[0]

    @classmethod
    def create_symbol(cls):
        cls._count += 1
        return MatrixSymbol('%s%d'%(cls._name, cls._count))

class ProductMultivariatePSpace(ProductPSpace, MultivariatePSpace):
    @property
    def density(self):
        symbol = BlockMatrix([space.symbol for space in self.spaces])
        mean = BlockMatrix([space.mean for space in self.spaces])
        covar = BlockDiagMatrix(*[space.covariance for space in self.spaces])

        return Tuple(symbol, mean, covar)

    def compute_density(self, expr, **kwargs):
        return MultivariatePSpace.compute_density(self, expr, **kwargs)

class RandomMatrixSymbol(RandomSymbol, MatrixSymbol):
    @property
    def shape(self):
        return self.symbol.shape
