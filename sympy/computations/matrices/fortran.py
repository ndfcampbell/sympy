from sympy.utilities.iterables import sift as groupby
from sympy.matrices.expressions import MatrixExpr, ZeroMatrix
from sympy.core import Expr

def is_number(x):
    return (isinstance(x, (int, float)) or
            isinstance(x, Expr) and x.is_Number)

def remove_numbers(coll):
    return filter(lambda x: not is_number(x.expr), coll)

def nameof(var):
    if is_number(var.expr):
        return var.expr
    else:
        return var.token

def call(x, assumptions):
    args = x.op.arguments(x.inputs, x.outputs)
    codemap = x.op.codemap([i.expr for i in x.inputs],
                           [nameof(a) for a in args], 'd', assumptions)
    return x.op.fortran_template % codemap

def getintent(comp, var):
    if isinstance(var.expr, ZeroMatrix):
        return None
    if var in comp.inputs and var in comp.outputs:
        return 'inout'
    if var in comp.inputs:
        return 'in'
    if var in comp.outputs:
        return 'out'

def gettype(comp, var):
    return 'real*8'

def shapeof(var):
    if isinstance(var.expr, MatrixExpr):
        return var.expr.shape

def shape_str(shape):
    if shape[0] == 1:
        return "(%s)"%str(shape[1])
    elif shape[1] == 1:
        return "(%s)"%str(shape[0])
    else:
        return "(%s, %s)"%(str(shape[0]), str(shape[1]))

def comment(var):
    return '  !  ' + str(var.expr)

def declaration(comp, var):
    s = gettype(comp, var)
    intent = getintent(comp, var)
    if intent:
        s += ", intent(%s)" % intent
    s += " :: "
    s += nameof(var)
    shape = shapeof(var)
    if shape:
        s += shape_str(shape)
    s += comment(var)
    return s

def dimen_declaration(dimen):
    return "integer, intent(in) :: %s" % dimen

def header(name, args):
    return ("subroutine %(name)s(%(args)s)" % {
                'name': name,
                'args': ', '.join(args)} +
            "\nimplicit none")

def footer():
    return "RETURN\nEND\n"

def dimensions(tcomp):
    shapes = map(shapeof, tcomp.variables)
    return set((d for shape in shapes if shape for d in shape))

def gen_fortran(tcomp, assumptions, name = 'f'):
    """
    inputs:
        tcomp - a tokenized computation (see inplace.tokenize)
    """

    vars = filter(lambda x: not is_number(x.expr), tcomp.variables)
    dimens = filter(lambda x: not is_number(x), dimensions(tcomp))



    intent = lambda v: getintent(tcomp, v)
    intents = groupby(vars, intent)

    head = header(name, [a.token for a in (intents['in'] + intents['inout'] +
                                           intents['out'])]
                        + map(str, dimens))

    declarations = '\n'.join(map(dimen_declaration, dimens) +
                            [declaration(tcomp, a)
                                    for x in ('in', 'inout', 'out', None)
                                    for a in intents[x]])

    calls = '\n'.join([call(comp, assumptions) for comp in tcomp.toposort()])

    foot = footer()

    return '\n\n'.join((head, declarations, calls, foot))

def build(tcomp, assumptions, name = 'f', **kwargs):
    flags = '-lblas', '-llapack'
    _id = abs(hash((tcomp, name)))
    src = kwargs.pop('src', 'tmp.f90')
    mod = kwargs.pop('mod', 'mod'+str(_id))
    code = gen_fortran(tcomp, assumptions, name)
    f = open(src, 'w')
    f.write(code)
    f.close()
    compile(src, mod, flags)
    module = __import__(mod)
    return module.__dict__[name]


def compile(src, mod, flags):
    """ Compile src file to module with flags using f2py """
    import os
    flagstr = ' '.join(flags)
    command = 'f2py -c %(src)s -m %(mod)s %(flagstr)s' % locals()
    if os.path.exists(mod+'.so'):
        return
    file = os.popen(command)
    s = file.read()
    if 'error' in s:
        print s
        raise Exception("File %s did not compile" % src)
