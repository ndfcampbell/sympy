from sympy import MatrixExpr, Expr, ZeroMatrix
from sympy.computations.core import Computation, unique
from sympy.computations.inplace import IOpComp, ExprToken
from sympy.utilities.iterables import sift
from functools import partial
from sympy.computations.matrices.fftw import FFTW

def groupby(key, coll):
    return sift(coll, key)

def remove(pred, coll):
    return [x for x in coll if not pred(x)]

with open('sympy/computations/matrices/fortran_template.f90') as f:
    template = f.read()

class FortranPrintableIOpComp(object):
    def fortran_footer(self, *args):
        return self.comp.fortran_footer(*args)
    def fortran_header(self, *args):
        return self.comp.fortran_header(*args)
    def fortran_use_statements(self):
        return self.comp.fortran_use_statements()


class FortranPrintableComputation(object):

    # DAG Functions
    def fortran_header(self, name, inputs, outputs):
        return 'subroutine %s(%s)'%(name, ', '.join(inputs+outputs))

    def fortran_use_statements(self):
        uses = []
        if self.has(FFTW):
          uses.append("use, intrinsic :: iso_c_binding")
        return join(uses)

    def fortran_include_statements(self):
        includes = []
        if self.has(FFTW):
          includes.append("include 'fftw3.f03'")
        return join(includes)


    def fortran_footer(self, name):
        return 'end subroutine %s'%(name)

    # Atomic Computation Functions
    def fortran_call(self, input_names, output_names):
        # TODO
        return '%s = %s(%s)'%(', '.join(output_names),
                              self.__class__.__name__,
                              ', '.join(input_names))

    def fortran_function_interface(self):
        return ''
    def fortran_function_definition(self):
        return ''


def update_class(old, new):
    for k, v in new.__dict__.items():
        if '__' not in k:
            setattr(old, k, v)

update_class(Computation, FortranPrintableComputation)
update_class(IOpComp, FortranPrintableIOpComp)

def join(L):
    return '  ' + '\n  '.join([x for x in L if x])

def generate_fortran(comp, inputs, outputs, types, name='f'):
    """ Generate Fortran code from a computation

    comp - a tokenized computation from inplace_compile
    inputs  - a list of SymPy (Matrix)Expressions
    outputs - a list of SymPy (Matrix)Expressions
    types   - a dictionary mapping expressions in your computation to types
    name    - the name of your subroutine
    """


    computations = comp.toposort()
    vars = list(comp.variables)

    input_tokens  = sorted_tokens(comp.inputs, inputs)
    output_tokens = sorted_tokens(comp.outputs, outputs)
    tokens = list(set(map(gettoken, vars)))
    dimens = dimensions(comp)

    function_definitions = join([c.comp.fortran_function_definition()
                                            for c in computations])
    subroutine_header = comp.fortran_header(name, input_tokens, output_tokens)

    use_statements = comp.fortran_use_statements()
    include_statements = comp.fortran_include_statements()

    function_interfaces = join([c.comp.fortran_function_interface()
                                            for c in computations])
    argument_declarations = ''

    argument_declarations = join([
        declare_variable(token, comp, types, inputs, outputs)
        for token in unique(input_tokens + output_tokens)])

    variable_declarations = join([
        declare_variable(token, comp, types, inputs, outputs)
        for token in (set(tokens) - set(input_tokens + output_tokens))])

    dimen_inits = map(dimension_initialization,
                      dimens,
                      map(partial(var_that_uses_dimension, vars=vars), dimens))
    variable_initializations = join(map(initialize_variable, vars)
                                  + dimen_inits)

    statements = join([
        c.comp.fortran_call(c.input_tokens, c.output_tokens)
        for c in computations])

    variable_destructions = join(map(destroy_variable, vars))

    footer = comp.fortran_footer(name)

    return template % locals()


gettoken = lambda x: x.token
def sorted_tokens(source, exprs):
    vars = sorted([v for v in source if v.expr in exprs],
                            key=lambda v: list(exprs).index(v.expr))
    return map(gettoken, vars)


#####################
# Variable Printing #
#####################

def shape_str(shape):
    """ Fortran string for a shape.  Remove 1's from Python shapes """
    if shape[0] == 1:
        return "(%s)"%str(shape[1])
    elif shape[1] == 1:
        return "(%s)"%str(shape[0])
    else:
        return "(%s, %s)"%(str(shape[0]), str(shape[1]))

def intent_str(isinput, isoutput):
    if isinput and isoutput:
        return ', intent(inout)'
    elif isinput and not isoutput:
        return ', intent(in)'
    elif not isinput and isoutput:
        return ', intent(out)'
    else:
        return ''

def declare_variable(token, comp, types, inputs, outputs):
    isinput  = any(token == v.token for v in comp.inputs if not
            constant_arg(v.expr))
    isoutput = any(token == v.token for v in comp.outputs if not
            constant_arg(v.expr) and v.expr in outputs)
    exprs = set(v.expr for v in comp.variables if v.token == token
                                          and not constant_arg(v.expr))
    if not exprs:
        return ''
    expr = exprs.pop()
    typ = types[expr]
    return declare_variable_string(token, expr, typ, isinput, isoutput)


def declare_variable_string(token, expr, typ, is_input, is_output):
    intent = intent_str(is_input, is_output)
    rv = typ + intent + ' :: ' + token
    if isinstance(expr, MatrixExpr):
        rv += shape_str(expr.shape)
    return rv

def initialize_variable(v):
    return ''

def destroy_variable(v):
    return ''

def is_number(x):
    return (isinstance(x, (int, float)) or
            isinstance(x, Expr) and x.is_Number)

def constant_arg(arg):
    """ Is this argument a constant?

    If so we don't want to include it as a parameter """
    return is_number(arg) or isinstance(arg, ZeroMatrix)

def dimension_declaration(dimen):
    return "integer :: %s" % str(dimen)

def dimension_initialization(dimen, var):
    return str(dimen) + ' = size(%s, %d)'%(var.token,
            var.expr.shape.index(dimen)+1)

def var_that_uses_dimension(dimen, vars):
    return next(v for v in vars if v.expr.has(dimen))

def dimensions(comp):
    """ Collect all of the dimensions in a computation

    For example if a computation contains MatrixSymbol('X', n, m) then n and m
    are in the set returned by this function """
    return set(remove(constant_arg, sum([v.expr.shape for v in comp.variables
                           if isinstance(v.expr, MatrixExpr)], ())))