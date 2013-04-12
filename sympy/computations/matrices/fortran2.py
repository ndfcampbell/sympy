from sympy import MatrixExpr, Expr, ZeroMatrix
from sympy.computations.core import Computation

with open('sympy/computations/matrices/fortran_template.f90') as f:
    template = f.read()

class FortranPrintableComputation(object):

    # DAG Functions
    def fortran_header(self, name, inputs, outputs):
        return 'subroutine %s(%s)'%(name, ', '.join(inputs+outputs))

    def fortran_use_statements(self):
        return ''

    def fortran_footer(self):
        return ''

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

def join(L):
    return '\n'.join([x for x in L if x])

def generate_fortran(comp, inputs, outputs, types, assumptions, name='f'):

    computations = comp.toposort()
    vars = comp.variables
    input_vars = [v for v in comp.inputs  if v.expr in inputs]
    output_vars = [v for v in comp.outputs if v.expr in outputs]
    input_tokens = map(lambda x: x.token, input_vars)
    output_tokens = map(lambda x: x.token, output_vars)


    function_definitions = join([c.comp.fortran_function_definition()
                                            for c in computations])
    subroutine_header = comp.fortran_header(name, input_tokens, output_tokens)

    use_statements = comp.fortran_use_statements()

    function_interfaces = join([c.comp.fortran_function_interface()
                                            for c in computations])

    variable_declarations = join([
        declare_variable(v, input_vars, output_vars, types) for v in vars
        if not constant_arg(v.expr)])

    variable_initializations = join(map(initialize_variable, vars))

    statements = join([
        c.comp.fortran_call(c.input_tokens, c.output_tokens)
        for c in computations])

    variable_destructions = join(map(destroy_variable, vars))

    footer = comp.comp.fortran_footer()

    return template % locals()


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

def declare_variable(v, input_vars, output_vars, types):
    typ = types[v.expr]
    intent = intent_str(v in input_vars, v in output_vars)
    rv = typ + intent
    if isinstance(v.expr, MatrixExpr):
        rv += ' :: ' + shape_str(v.expr.shape)
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
