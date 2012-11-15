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

def make_idinc():
    cache = {}
    def idinc(x):
        id = cache.get(x, 0) + 1
        cache[x] = id
        return id
    return idinc

def copies_one(comp, idinc=make_idinc()):
    return CompositeComputation(
                           *[CopyComp(comp.inputs[idx], idinc(comp.inputs[idx]))
                                for idx in inplace(comp).values()])

def purify_one(comp, idinc=make_idinc()):
    return comp + copies_one(comp, idinc)

class ExprToken(Basic):
    expr = property(lambda self: self.args[0])
    token = property(lambda self: self.args[1])

class OpComp(Computation):
    op = property(lambda self: self.args[0])
    inputs = property(lambda self: self.args[1])
    outputs = property(lambda self: self.args[2])
    inplace = property(lambda self: self.op.inplace)

def naive_tokenize_one(mathcomp, tokenizer = make_getname()):
    return OpComp(type(mathcomp),
                  tuple(ExprToken(i, tokenizer(i)) for i in mathcomp.inputs),
                  tuple(ExprToken(o, tokenizer(o)) for o in mathcomp.outputs))
