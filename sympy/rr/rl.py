# Generic rules for SymPy
from sympy import Basic

def rmid(isid):
    """ Create a rule to remove identities

    isid - fn :: x -> Bool  --- whether or not this element is an identity
    """
    def ident_remove(expr):
        """ Remove identities """
        ids = map(isid, expr.args)
        if sum(ids) == 0:           # No identities. Common case
            return expr
        elif sum(ids) != len(ids):  # there is at least one non-identity
            return Basic.__new__(expr.__class__,
                              *[arg for arg, x in zip(expr.args, ids) if not x])
        else:
            first_id = (arg for arg, x in zip(expr.args, ids) if x).next()
            return Basic.__new__(expr.__class__, first_id)

    return ident_remove

def frequencies(coll):
    counts = {}
    for elem in coll:
        counts[elem] = counts.get(elem, 0) + 1
    return counts

def glom(mkglom):
    """ Create a rule to conglomerate identical args

    >>> from sympy.rr import glom
    >>> rl = glom(lambda num, arg: num * arg)
    >>> rl(Basic(1, 1, 3))
    Basic(2, 3)
    """
    def conglomerate(expr):
        """ Conglomerate together identical args x + x -> 2x """
        freqs = frequencies(expr.args)
        return Basic.__new__(type(expr), *[arg if freqs[arg] == 1
                                               else mkglom(freqs[arg], arg)
                                               for arg in freqs])
    return conglomerate

def unpack(expr):
    if len(expr.args) == 1:
        return expr.args[0]
    else:
        return expr

def flatten(expr):
    """ Flatten T(a, b, T(c, d), T2(e)) to T(a, b, c, d, T2(e)) """
    cls = expr.__class__
    args = []
    for arg in expr.args:
        if arg.__class__ == cls:
            args.extend(arg.args)
        else:
            args.append(arg)
    return Basic.__new__(expr.__class__, *args)