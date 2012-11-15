from sympy import Basic
from sympy.computations.core import Computation, CompositeComputation

def make_getname():
    cache = {}
    seen = set(['', None])

    def getname(key, requested=None):
        """ Get the name associated to a key """
        if key in cache:
            return cache[key]

        name = requested
        if not name and hasattr(key, 'name'):
            name = key.name
        else:
            name = ''
        if name in seen:
            id = 2
            while(name + '_%d'%id in seen):
                id += 1
            name = name + '_%d'%id

        cache[key] = name
        seen.add(name)
        return name
    return getname


class Copy(Basic):
    arg = property(lambda self: self.args[0])
    tag = property(lambda self: self.args[1])

    name = property(lambda self: self.arg.name if hasattr(self.arg, 'name')
                                               else 'copy')

def inplace(x):
    try:
        return x.inplace
    except AttributeError:
        return {}

class CopyComp(Computation):
    tag = property(lambda self: self.args[1])
    inputs = property(lambda self: (self.args[0],))
    outputs = property(lambda self: (Copy(self.args[0], self.tag),))

def purify_one(comp, numseq=iter(xrange(1, int(1e9)))):
    return CompositeComputation(comp, *[CopyComp(comp.inputs[idx], id)
        for idx, id in zip(inplace(comp).values(), numseq)])
