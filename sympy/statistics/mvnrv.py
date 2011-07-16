from rv import Domain, SingleDomain, PSpace
from sympy import Interval, S, initeSet, Symbol, Tuple
from sympy.matrices import BlockMatrix, BlockDiagMatrix, linear_factors, Transpose

R = Interval(-oo, oo)

def is_linear(expr, syms=None):
    return True

class MultivariateDomain(Domain):
    is_MultiVariate = True

class SingleMultivariateDomain(MultivariateDomain, SingleDomain):
    def __new__(cls, symbol):
        assert symbol.is_Symbol
        symbols = FiniteSet(symbol)
        return Domain.__new__(cls, symbols, R**symbol.n)

    def as_boolean(self):
        return self.set.as_relational(self.symbol)

class ConditionalMultivariateDomain(MultivariateDomain, ConditionalDomain):
    pass

class MultivariatePSpace(PSpace):
    is_MultiVariate = True

    @property
    def symbol(self):
        return self.density[0]
    @property
    def mean(self):
        return self.density[1]
    @property
    def covariance(self):
        return self.density[2]

    def integrate(self, expr, rvs=None, **kwargs):
        if rvs == None:
            rvs = self.values
        else:
            rvs = frozenset(rvs)

        assert is_linear(expr, rvs)

        #return expr.subs{rv:mean_of_rv}

        return None

    def compute_density(self, expr, **kwargs):

        expr = expr.subs({rv:rv.symbol for rv in self.values})

        d = linear_factors(expr, self.symbols)

        # Construct row vector
        sym = self.symbol
        rowvec = Blockmatrix([[d[sym] for sym in self.symbol]])

        mean = rowvec * self.mean
        covar = rowvec * self.covariance * Transpose(rowvec)

        return mean, covar

    def P(self, condition, **kwargs):
       raise NotImplementedError(
               "Operation not implemented for Multivariate case")

    def where(self, condition):
       raise NotImplementedError(
               "Operation not implemented for Multivariate case")

    def conditional_space(self, condition, **kwargs):

        condition = condition.subs({rv:rv.symbol for rv in self.values})

        domain = ConditionalMultivariateDomain(self.domain, condition)
        density = self.density

        return ContinuousPSpace(domain, density)



class SingleMultivariatePSpace(MultivariatePSpace):
    _count = 0
    _name = 'X'
    def __new__(cls, symbol, mean, covariance):
        assert symbol.is_MatrixSymbol
        domain = SingleMultivariateDomain(symbol)
        density = Tuple(symbol, mean, covariance)
        return MultivariatePSpace.__new__(cls, domain, density)

    @property
    def value(self):
        return tuple(self.values)[0]

class ProductContinuousPSpace(ProductPSpace, ContinuousPSpace):
    @property
    def density(self):
        symbol = BlockMatrix([[space.symbol] for space in self.spaces])
        mean = BlockMatrix([[space.mean] for space in self.spaces])
        covar = BlockDiagMatrix([space.covariance for space in self.spaces])

        return Tuple(symbol, mean, covar)
