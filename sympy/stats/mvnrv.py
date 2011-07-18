from rv import (Domain, SingleDomain, PSpace, ConditionalDomain, ProductPSpace,
        RandomSymbol)
from sympy import Interval, S, FiniteSet, Symbol, Tuple
from sympy.matrices import (BlockMatrix, BlockDiagMatrix, linear_factors,
        Transpose, MatrixSymbol, block_collapse)


oo = S.Infinity
R = Interval(-oo, oo)
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
        if rvs == None:
            rvs = self.values
        else:
            rvs = frozenset(rvs)

        assert is_linear(expr, rvs)

        #return expr.subs{rv:mean_of_rv}

        return None

    def compute_density(self, expr, **kwargs):

        expr = expr.subs(dict((rv, rv.symbol) for rv in self.values))

        d = linear_factors(expr, *self.symbols)

        # Construct row vector
        # Be careful because we may have blocks of blockmatrices in self.symbol
        def access_d(sym):
            if sym.is_Symbol:
                return d[sym] # Just return the coefficient
            if sym.is_BlockMatrix: # need to recursively return BlockMatrix
                return BlockMatrix([[access_d(s) for s in sym]])

        if self.symbol.is_BlockMatrix:
            # Operator is a block row vector
            operator = BlockMatrix([[access_d(sym) for sym in self.symbol]])
        else:
            operator = d[self.symbol]

        additive_terms = expr - block_collapse(operator * self.symbol)[0,0]
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

        condition = condition.subs(dict((rv, rv.symbol) for rv in self.values))

        domain = ConditionalMultivariateDomain(self.domain, condition)
        density = self.density

        return MultivariatePSpace(domain, density)



class SingleMultivariatePSpace(MultivariatePSpace):
    _count = 0
    _name = 'X'
    def __new__(cls, symbol, mean, covariance):
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

class ProductMultivariatePSpace(ProductPSpace, MultivariatePSpace):
    @property
    def density(self):
        symbol = BlockMatrix([space.symbol for space in self.spaces])
        mean = BlockMatrix([space.mean for space in self.spaces])
        covar = BlockDiagMatrix([space.covariance for space in self.spaces])

        return Tuple(symbol, mean, covar)

    def compute_density(self, expr, **kwargs):
        return MultivariatePSpace.compute_density(self, expr, **kwargs)


class RandomMatrixSymbol(RandomSymbol, MatrixSymbol):

    @property
    def shape(self):
        return self.symbol.shape
