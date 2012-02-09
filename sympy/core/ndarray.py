
from sympy import *
def isiterable(x):
    try:
        iter(x)
        return True
    except:
        return False
def strides(l):
    numplaces = 1
    s = []
    for i in reversed(l):
        s.append(numplaces)
        numplaces *= i
    return s[::-1]

class NDArray(object):

    @property
    def shape(self):
        raise NotImplementedError()
    def __getitem__(self, key):
        self.assert_in_bounds(key)
        return self._getitem(key)
        raise NotImplementedError()

    def assert_in_bounds(self, key):
        if not all(0<=i<dim for i,dim in zip(key, self.shape)):
            raise IndexError("Index %s out of range"%str(key))

class DenseNDArray(NDArray):
    def __new__(cls, shape, storage):
        assert len(storage) == prod(shape)
        shape = Tuple(*shape)
        storage = cls.create_storage(storage)
        if isinstance(storage, Basic):
            return Basic.__new__(cls, shape, storage)
        else:
            obj = object.__new__(cls)
            obj.args = (shape, storage)
            return obj

    @property
    def shape(self):
        return self.args[0]
    @property
    def storage(self):
        return self.args[1]

    def index(self, key):
        return sum([i*stride for i,stride in zip(key, strides(self.shape))])

    def _getitem(self, key):
        self.assert_in_bounds(key)
        if not isiterable(key): key = (key,)

        return self.storage[self.index(key)]

    def __setitem__(self, key, value):
        self.assert_in_bounds(key)
        self.storage[self.index(key)] = value

class DenseNDArrayTuple(DenseNDArray, Basic):
    @classmethod
    def create_storage(cls, storage):
        return Tuple(*storage)

class DenseNDArrayList(DenseNDArray):
    @classmethod
    def create_storage(cls, storage):
        return list(storage)

try:
    import numpy as np
    class DenseNDArrayNumpy(DenseNDArray):
        def __new__(cls, shape, storage):
            shape = Tuple(*shape)
            storage = np.asarray(storage)
            obj = object.__new__(cls)
            obj.args = (shape, storage.flat)
            obj._array_storage = storage.reshape(shape)
            return obj

        @classmethod
        def create_storage(cls, storage):
            return np.asarray(storage)

        def __getitem__(self, key):
            return self._array_storage.__getitem__(key)
except: pass

class SparseNDArray(NDArray):
    pass

class SparseNDArrayDict(SparseNDArray):
    pass

class FunctionalNDArray(NDArray, Basic):
    def __new__(cls, shape, fn):
        assert callable(fn)
        shape = Tuple(*shape)
        return Basic.__new__(cls, shape, fn)

    @property
    def shape(self):
        return self.args[0]
    @property
    def _fn(self):
        return self.args[1]

    def _getitem(self, key):
        return self._fn(*key)



