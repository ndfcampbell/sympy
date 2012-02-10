from tensor import *
class LinearOperator(Tensor):
    @property
    def shape(self):
        return self.sizes[0][0], self.sizes[1][0]

class LinearOperatorSymbol(TensorSymbol, LinearOperator):
    def __new__(cls, name, rows=None, cols=None):
        if rows and rows>1 : contra_sizes = (rows,)
        else:                contra_sizes = ()
        if cols and cols>1 : covar_sizes  = (cols,)
        else:                covar_sizes  = ()

        return TensorSymbol.__new__(cls, name, (contra_sizes, covar_sizes))

class VectorSymbol(LinearOperatorSymbol)
    def __new__(cls, name, rows):
        return LinearOperatorSymbol.__new__(cls, name, rows, None)

class DualVectorSymbol(LinearOperatorSymbol):
    def __new__(cls, name, cols):
        return LinearOperator.__new__(cls, name, None, cols)
    def __getitem__(self, col_index):
        return TensorSymbol.__getitem__(self, (None, col_index))

class MetricTensor(Tensor):
    def __call__(self, a, b):
        i,j = indices('ij')
        return self[None, i|j]*a[i]*b[j]
    def lower(self, d):
        pass
    def raise(self, v):
        pass

class MetricTensorSymbol(TensorSymbol, MetricTensor):
    def __new__(cls, name, dimensions):
        return TensorSymbol.__new__(name, ((), (dimensions, dimensions)))

