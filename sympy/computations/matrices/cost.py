from refflops import flopfuncs
from sympy.matrices.expressions import MatrixExpr
from sympy import Expr

typebytes = {'D': 8, 'S': 4, 'Z': 16, 'C': 8}

def flops(comp):
    """ Compute the flops of a computation

    See also refflops.py"""
    name = type(comp).__name__.lower()
    return flopfuncs[name](*comp.raw_inputs)


def numelements(expr):
    """ Number of elements (ints, doubles, etc...) or an expression

    >>> numelements(MatrixSymbol('A', 2, 3))
    6
    >>> numelements(Symbol('x'))
    1
    """
    if isinstance(expr, MatrixExpr):
        return expr.shape[0] * expr.shape[1]
    if isinstance(expr, Expr):
        return 1
    raise TypeError()


def nbytesofoutputs(comp):
    """ Number of bytes of the inputs of a computation """
    return sum(map(numelements, comp.outputs)) * typebytes[comp.typecode]


def nbytesofinputs(comp):
    """ Number of bytes of the outputs of a computation """
    return sum(map(numelements, comp.raw_inputs)) * typebytes[comp.typecode]


def commcost(comp, A, B, getdata):
    """ Time to communicate results of ``comp`` from ``A`` to ``B`` """
    latency, bandwidth = getdata(A, B)
    return latency + nbytesofoutputs(comp) / bandwidth


def memtime(nbytes, memhierarchy):
    """ Time to load ``nbytes`` from memory

    memhierarchy - list of bandwidth (bytes/s), size (bytes) pairs

    >>> memhierarchy = [(1e9, 1e3), (1e8, 1e6), (1e6, 4e9)] # regs, L1, mainmem
    >>> memtime(10000, memhierarchy)  # registers then some cache
    9e-05
    """
    time = 0
    for bandwidth, size in memhierarchy:
        time = time + min(size, nbytes) / float(bandwidth)
        nbytes = nbytes - min(size, nbytes)
        if not nbytes:
            break
    return time

def compcost(comp, A, getdata):
    """ Time to compute ``comp`` on ``A`` """
    timeperflop, memhierarchy = getdata(A)
    computetime = flops(comp) / timeperflop
    memorytime = memtime(nbytesofinputs(comp), memhierarchy)
    return max(memorytime, computetime)
