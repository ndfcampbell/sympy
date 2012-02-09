from sympy import *

class Tensor(Basic):
    @property
    def sizes(self):
        raise NotImplementedError()

    @property
    def rank(self):
        return tuple(map(len, self.sizes))

    def __getitem__(self, key):
        if isinstance(key, IndexSet):
            contravariants = key
            covariants     = None
        else:
            contravariants, covariants = key
        contravariants  = contravariants or EmptyIndexSet
        covariants      = covariants or EmptyIndexSet
        assert (len(contravariants), len(covariants)) == self.rank, "Bad Rank"
        return self._index(contravariants, covariants)

    def _index(self, contravariants, covariants):
        return IndexedTensor(self, contravariants, covariants)

    @property
    def display(self):
        raise NotImplementedError()
    def __str__(self):
        raise NotImplementedError()
    __repr__ = __str__

class IndexedTensor(Expr):
    def __new__(cls, tensor, contravariants, covariants):
        assert (len(contravariants), len(covariants)) == tensor.rank, "Bad Rank"
        return Basic.__new__(cls, tensor, contravariants, covariants)

    @property
    def tensor(self):
        return self.args[0]
    @property
    def contravariants(self):
        return self.args[1]
    @property
    def covariants(self):
        return self.args[2]

    @property
    def dimension_dict(self):
        d = dict()
        for index, size in zip(self.contravariants, self.tensor.sizes[0]):
            if index in d : assert d[index] == size
            else          : d[index] = size
        for index, size in zip(self.covariants, self.tensor.sizes[1]):
            if index in d : assert d[index] == size
            else          : d[index] = size
        return d

    def free_indices(self):
        return (FiniteSet(*self.contravariants) - FiniteSet(*self.covariants),
                FiniteSet(*self.covariants) - FiniteSet(*self.contravariants))
    @property
    def free_contravariants(self):
        return self.free_indices()[0]
    @property
    def free_covariants(self):
        return self.free_indices()[1]
    @property
    def rank(self):
        return len(self.free_contravariants), len(self.free_covariants)

    def __str__(self):
        s = str(self.tensor)
        if len(self.contravariants)!=0:
            s += '^%s'%str(self.contravariants)
        if len(self.covariants)!=0:
            s += '_%s'%str(self.covariants)

        return s

    __repr__ = __str__

    def __mul__(self, other):
        return TensorMul(self, other)

class TensorSymbol(Tensor):
    def __new__(cls, name, sizes):
        return Basic.__new__(cls, name, sizes)

    @property
    def name(self):
        return self.args[0]
    @property
    def sizes(self):
        return self.args[1]

    @property
    def display(self):
        return self.name

    def __str__(self):
        return self.name

class IndexSet(Basic):
    def __new__(cls, *indices):
        assert all(isinstance(i, IndexSet) for i in indices)
        new_indices = []
        for i in indices:
            if len(i)>1:
                new_indices.extend(i.args)
            else:
                new_indices.append(i)
        assert all(isinstance(i, IndexSet) for i in new_indices)
        return Basic.__new__(cls, *new_indices)

    @property
    def indices(self):
        return self.args
    def join(self, other):
        return IndexSet(*(self.indices + other.indices))
    __or__ = join

    def __len__(self):
        return len(self.indices)
    def __iter__(self):
        return iter(self.indices)

    def __str__(self):
        return ''.join(map(str,self.indices))

EmptyIndexSet = IndexSet()

class Index(Symbol, IndexSet):
    @property
    def indices(self):
        return (self,)
    __or__ = IndexSet.__or__

    def __str__(self):
        return self.name

def indices(s):
    return map(Index, s)

class TensorMul(IndexedTensor):

    def __new__(cls, *indexed_tensors):
        assert all(isinstance(it,IndexedTensor) for it in indexed_tensors)
        T = Basic.__new__(cls, *indexed_tensors)
        T.dimension_dict # call this to through assertions as a side-effect
        return T

    @property
    def dimension_dict(self):
        d = dict()
        for it in self.args:
            for index,size in it.dimension_dict.items():
                if index in d : assert d[index] == size, "Bad Shape"
                else          : d[index] = size

        return d

    @property
    def contravariants(self):
        return IndexSet(*[ind for arg in self.args for ind in
                arg.contravariants.indices])
    @property
    def covariants(self):
        return IndexSet(*[ind for arg in self.args for ind in
                arg.covariants.indices])

    def __str__(self):
        return '*'.join(map(str, self.args))

def MatrixSymbol(name, rows=None, cols=None):
    if rows and rows>1 : contra_sizes  = (rows,)
    else:                contra_sizes= ()
    if cols and cols>1 :  covar_sizes = (cols,)
    else:                 covar_sizes = ()

    return TensorSymbol(name, (contra_sizes, covar_sizes))
