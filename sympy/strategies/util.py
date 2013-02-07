from sympy import Basic
from sympy.utilities.iterables import sift

def groupby(fn, coll):
    return sift(coll, fn)

identity = lambda x: x

def count(tup):
    return dict((k, len(v)) for k,v in groupby(identity, tup).items())

new = Basic.__new__

def assoc(d, k, v):
    d = d.copy()
    d[k] = v
    return d

basic_fns = {'op': type,
             'new': Basic.__new__,
             'leaf': lambda x: not isinstance(x, Basic) or x.is_Atom,
             'children': lambda x: x.args}

expr_fns = assoc(basic_fns, 'new', lambda op, *args: op(*args))

class RuleDB(object):
    def __init__(self):
        self.coll = list()

    def insert(self, tup, val):
        self.coll.append((count(tup), val))

    def query(self, indata):
        inkey = count(indata)
        for collkey, collval in self.coll:
            if all(inkey.get(k, 0) >= v for k, v in collkey.items()):
                yield collval
