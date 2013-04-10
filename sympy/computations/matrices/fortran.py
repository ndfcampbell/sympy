from sympy.utilities.iterables import sift
from sympy.matrices.expressions import MatrixExpr, ZeroMatrix
from sympy.core import Expr, Symbol

def groupby(key, coll):
    return sift(coll, key)

def is_number(x):
    return (isinstance(x, (int, float)) or
            isinstance(x, Expr) and x.is_Number)

def nameof(var):
    """ Fortran name of variable """
    if is_number(var.expr):
        return var.expr
    else:
        return var.token

def call(x):
    """ Fortran string to compute x, a computation """
    args = x.op.arguments(x.inputs, x.outputs)
    codemap = x.op.codemap([i.expr for i in x.inputs],
                           [nameof(a) for a in args], 'd')
    return x.op.fortran_template % codemap

def intentsof(comp):
    """ All Fortran intents of a computation

    Returns:
        dictionary mapping Fortran variable name to intent
    """
    in_tokens = [v.token for v in comp.inputs if not constant_arg(v.expr)]
    out_tokens = [v.token for v in comp.outputs]

    def intentof_tok(token):
        if token in in_tokens and token in out_tokens:
            return 'inout'
        if token in in_tokens:
            return 'in'
        if token in out_tokens:
            return 'out'

    vars_by_token = groupby(lambda v: v.token, comp.variables)
    tokens = set([v.token for v in comp.variables])
    intents = dict(zip(tokens, map(intentof_tok, tokens)))
    intents2 = dict((tok, intent)
                    if not all(constant_arg(v) for v in vars_by_token[tok]) else
                    (tok, None)
                    for tok, intent in intents.items())
    return intents2

def gettype(comp, expr):
    """ The type of a mathematical expression in a computation

    TODO: makethis not trivial.  Atomic BLAS/LAPACK computations hold the
    dtype, we just need to collect it well """
    if expr == Symbol('INFO'):
        return 'integer'
    return 'real*8'

def shapeof(expr):
    if isinstance(expr, MatrixExpr):
        return expr.shape

def shape_str(shape):
    """ Fortran string for a shape.  Remove 1's from Python shapes """
    if shape[0] == 1:
        return "(%s)"%str(shape[1])
    elif shape[1] == 1:
        return "(%s)"%str(shape[0])
    else:
        return "(%s, %s)"%(str(shape[0]), str(shape[1]))

def comment(vars):
    """ Fortran comment string.  Prints X -> Y -> Z  for vars X,Y,Z """
    return '  !  ' + ' -> '.join([str(v.expr) for v in vars])

def getdeclarations(comp):
    """ Full declarations for each Fortran variable in computation

    Returns dict mapping Fortran variable name to declaration string """
    tokens = groupby(lambda v: v.token, comp.variables)
    tokens = {k: vs for k,vs in tokens.items()
                    if any(not is_number(v.expr) for v in vs)}
    intents = intentsof(comp)
    def declaration_string(tok):
        vars   = tokens[tok]
        var    = vars[0]
        expr   = var.expr
        type   = gettype(comp, expr)
        intent = intents[var.token]
        intentstr = ", intent(%s)" % intent if intent else ""
        name   = nameof(var)
        shape  = shapeof(expr)
        shapestr = shape_str(shape) if shape else ""
        cmnt = comment(vars)
        return ("%(type)s%(intentstr)s :: %(name)s%(shapestr)s%(cmnt)s" %
                locals())

    return dict(zip(tokens, map(declaration_string, tokens)))

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
    """ Collect all of the dimensions in a computation

    For example if a computation contains MatrixSymbol('X', n, m) then n and m
    are in the set returned by this function """
    shapes = [shapeof(v.expr) for v in tcomp.variables]
    return set((d for shape in shapes if shape for d in shape))

def unique_tokened_variables(vars):
    """ Given a collection of many ExprTokens select a representative sample

    This sample should include exacty one ExprToken for each token
    """

    return [vs[0] for tok, vs in groupby(lambda et: et.token, vars).items()]

intent_ranks = ['in', 'inout', 'out', None]

def sort_arguments(args, order=()):
    """ Sort arguments

    Sorts by the order in which expressions occur in order
    if variables' expressions aren't in order then they are last
    These variables are sorted lexicographically by token
    """
    args = (sorted(filter(lambda x: x.expr     in order, args),
                   key =  lambda x: order.index(x.expr)) +
            sorted(filter(lambda x: x.expr not in order, args),
                   key =  lambda x: str(x.token)))
    return args

def constant_arg(arg):
    """ Is this argument a constant?

    If so we don't want to include it as a parameter """
    return is_number(arg) or isinstance(arg, ZeroMatrix)

def remove(pred, coll):
    return filter(lambda x: not pred(x), coll)

def gen_fortran(tcomp, name = 'f', input_order=()):
    """
    inputs:
        tcomp - a tokenized computation (see inplace.tokenize)
    outputs:
        a string for a Fortran subroutine
    """

    tok_intents = intentsof(tcomp)
    intent = lambda x: tok_intents[x.token]
    vars = sorted(filter(lambda x: not is_number(x.expr), tcomp.variables),
                  key =  lambda x: intent_ranks.index(intent(x)))
    dimens = sorted(filter(lambda x: not is_number(x), dimensions(tcomp)),
                    key = str)

    intents = groupby(intent, unique_tokened_variables(vars))

    vars_by_token = groupby(lambda x: x.token, tcomp.variables)
    arguments = [tok for tok, intent in tok_intents.items() if intent]
    def key_func(tok):
        try:
            idx = max(input_order.index(v.expr) for v in vars_by_token[tok])
            return (1, idx)
        except ValueError:  # Token not in input_order
            return (2, str(tok))  # sort by string otherwise

    head = header(name, sorted(arguments, key=key_func)
                        + map(str, dimens))

    decs = getdeclarations(tcomp)
    sorted_tokens = sorted(decs.keys(),
            key = lambda tok: intent_ranks.index(tok_intents[tok]))
    declarations = '\n'.join(map(dimen_declaration, dimens) +
                             [decs[tok] for tok in sorted_tokens])

    calls = '\n'.join(map(call, tcomp.toposort()))

    foot = footer()

    return '\n\n'.join((head, declarations, calls, foot))

def build(tcomp, name='f', src='tmp.f90', mod=None,
            flags=['-lblas', '-llapack'], **kwargs):
    """ Build a Python function from SymPy Computation

    This function
    1.  generates fortran
    2.  writes it to the file ``src``
    3.  compiles it with ``flags`` (see ``compile``)
    4.  runs ``f2py`` on it (happens in ``compile``)
    5.  imports that function from the compiled module

    Returns:
        Python function
    """
    mod = mod or 'module_'+str(abs(hash((tcomp, name))))

    code = gen_fortran(tcomp, name, **kwargs)
    with open(src, 'w') as f:
        f.write(code)

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
