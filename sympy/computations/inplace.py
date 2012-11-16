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



def inplace(x):
    try:
        return x.inplace
    except AttributeError:
        try:
            return x.op.inplace
        except AttributeError:
            pass
    return {}

class Copy(Computation):
    pass

def make_idinc():
    cache = {}
    def idinc(x):
        id = cache.get(x, 0) + 1
        cache[x] = id
        return id
    return idinc

def copies_one(comp, getname=make_getname()):
    def new_comp(inp, out):
        newtoken = getname((inp.expr, out.expr), inp.token)
        out = ExprToken(inp.expr, newtoken)
        return OpComp(Copy, (inp,), (out,))

    return [new_comp(comp.inputs[v], comp.outputs[k])
                 for k, v in inplace(comp).items()]

def purify_one(comp, getname=make_getname()):
    copies = copies_one(comp, getname)
    d = dict((cp.inputs[0], cp.outputs[0]) for cp in copies)
    if not d:
        return comp

    inputs = tuple(d[i] if i in d else i for i in comp.inputs)

    newcomp = OpComp(comp.op, inputs, comp.outputs)  #.canonicalize() ??

    return CompositeComputation(newcomp, *copies)

class ExprToken(Basic):
    expr = property(lambda self: self.args[0])
    token = property(lambda self: self.args[1])

class OpComp(Computation):
    op = property(lambda self: self.args[0])
    inputs = property(lambda self: self.args[1])
    outputs = property(lambda self: self.args[2])
    inplace = property(lambda self: self.op.inplace)

    def __str__(self):
        ins  = "["+', '.join(map(str, self.inputs)) +"]"
        outs = "["+', '.join(map(str, self.outputs))+"]"
        return "%s -> %s -> %s"%(ins, str(self.op), outs)

def tokenize_one(mathcomp, tokenizer=make_getname()):
    return OpComp(type(mathcomp),
                  tuple(ExprToken(i, tokenizer(i)) for i in mathcomp.inputs),
                  tuple(ExprToken(o, tokenizer(o)) for o in mathcomp.outputs))

def tokenize(mathcomp, tokenizer=make_getname()):
    if not isinstance(mathcomp, CompositeComputation):
        return tokenize_one(mathcomp, tokenizer)
    return CompositeComputation(*map(tokenize_one, mathcomp.computations))
