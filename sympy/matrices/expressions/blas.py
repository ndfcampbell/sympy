from sympy.computations import InplaceComputation, CompositeComputation
from sympy import Basic, Tuple, Symbol, Q, symbols, ask
from sympy.matrices.expressions import MatrixSymbol, Transpose
from sympy.rules.tools import subs
from sympy.utilities.iterables import merge

basetypes = {'S': 'real*4', 'D': 'real*8', 'C': 'complex*8', 'Z': 'complex*16'}

class MatrixComputation(InplaceComputation):
    name = 'f'

    def shapes(self):
        return {x: x.shape for x in self.variables if hasattr(x, 'shape')}


    def dimensions(self):
        return set(d for x, shape in self.shapes().items()
                     for d in shape if isinstance(d, Symbol))

    def declarations(self, namefn):
        inplace = self.inplace_fn()(self)
        def declaration(x):
            s = "%s" % inplace.types()[x]
            if x in inplace.intents():
                s += ", intent(%s)" % inplace.intents()[x]
            s += " :: "
            s += "%s" % namefn(x)
            if x in inplace.shapes():
                s += "%s" % str(inplace.shapes()[x])
            return s
        return map(declaration,
                sorted(set(inplace.variables) | set(inplace.dimensions()), key=str))

    def intents(self):
        def intent(x):
            if x in self.inputs and x in self.replacements().values():
                return 'inout'
            if x in self.inputs or x in self.dimensions():
                return 'in'
            if x in self.outputs:
                return 'out'

        return {x: intent(x) for x in set(self.variables) | self.dimensions()
                             if intent(x)}

    def header(self, namefn):
        return "subroutine %(name)s(%(inputs)s)" % {
            'name': self.name,
            'inputs': ', '.join(map(namefn, self.inputs+tuple(self.dimensions())))}

    def footer(self):
        return "RETURN\nEND\n"

    def print_Fortran(self, namefn, assumptions=True):
        return '\n\n'.join([self.header(namefn),
                            '\n'.join(self.declarations(namefn)),
                            '\n'.join(self.calls(namefn, assumptions)),
                            self.footer()])

    def write(self, filename, *args, **kwargs):
        file = open(filename, 'w')
        file.write(self.print_Fortran(*args, **kwargs))
        file.close()

    def build(self, *args, **kwargs):
        import os
        src = kwargs.get('src', 'tmp.f90')
        mod = kwargs.get('mod', 'blasmod')
        self.write(src, *args, **kwargs)

        command = 'f2py -c %(src)s -m %(mod)s -lblas' % locals()
        file = os.popen(command); file.read()
        module = __import__(mod)
        return module.__dict__[self.name]

class BLAS(MatrixComputation):
    # TODO: metaclass magic for s/d/z prefixes
    def __new__(cls, *inputs, **kwargs):
        typecode = kwargs.get('typecode', 'D')
        mapping = dict(zip(cls._inputs, inputs))
        outputs = subs(mapping)(Tuple(*cls._outputs))
        return Basic.__new__(cls, Tuple(*inputs),
                                  Tuple(*outputs),
                                  typecode)

    def types(self):
        return merge({v: basetypes[self.typecode] for v in self.variables},
                     {d: 'integer' for d in self.dimensions()})

    @property
    def typecode(self):
        return self.args[2]

    def calls(self, namefn, assumptions=True):
        return [self.fortran_template % self.codemap(namefn, assumptions)]

    @property
    def variables(self):
        return self.inputs + self.outputs

    def fnname(self):
        """ GEMM(...).fnname -> dgemm """
        return (self.typecode+self.__class__.__name__).lower()

alpha = Symbol('alpha')
beta  = Symbol('beta')
n,m,k = symbols('n,m,k')
A = MatrixSymbol('A', n, m)
B = MatrixSymbol('B', m, k)
C = MatrixSymbol('C', n, k)
S = MatrixSymbol('S', n, n)
x = MatrixSymbol('x', n, 1)
a = MatrixSymbol('a', m, 1)
b = MatrixSymbol('b', k, 1)

class MM(BLAS):
    _inputs   = (alpha, A, B, beta, C)
    _outputs  = (alpha*A*B + beta*C,)
    view_map  = {0: 4}
    condition = True

    def codemap(self, namefn, assumptions=True):
        varnames = 'alpha A B beta C'.split()
        alpha, A, B, beta, C = self.inputs
        names    = map(namefn, (alpha, detranspose(A), B, beta, C))
        namemap  = dict(zip(varnames, names))
        other = {'TRANSA': trans(A), 'TRANSB': trans(B),
                 'LDA': LD(A), 'LDB': LD(B), 'LDC': LD(C),
                 'M':str(C.shape[0]), 'K':str(B.shape[0]), 'N':str(C.shape[1]),
                 'fn': self.fnname(),
                 'SIDE': left_or_right(A, B, Q.symmetric, assumptions),
                 'DIAG': diag(A, assumptions),
                 'UPLO': 'U'} # TODO: symmetric matrices might be stored low
        return merge(namemap, other)

class GEMM(MM):
    fortran_template = ("call %(fn)s('%(TRANSA)s', '%(TRANSB)s', "
                        "%(M)s, %(N)s, %(K)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, "
                        "%(B)s, %(LDB)s, %(beta)s, %(C)s, %(LDC)s)")

class SYMM(MM):
    condition = Q.symmetric(A) | Q.symmetric(B)
    fortran_template = ("call %(fn)s('%(SIDE)s', '%(UPLO)s', %(M)s, %(N)s, "
                        "%(alpha)s, %(A)s, %(LDA)s, %(B)s, %(LDB)s, "
                        "%(beta)s, %(C)s, %(LDC)s)")

class TRMM(MM):
    _inputs = (alpha, A, B)
    _outputs = (alpha*A*B,)
    view_map  = {0: 2}
    condition = Q.triangular(A) | Q.triangular(B)
    fortran_template = ("call %(fn)s('%(SIDE)s', '%(UPLO)s', '%(TRANSA)s', "
                        "'%(DIAG)s', %(M)s, %(N)s, %(alpha)s, %(A)s, %(LDA)s, "
                        "%(B)s, %(LDB)s)")
    def codemap(self, namefn, assumptions=True):
        varnames = 'alpha A B'.split()
        alpha, A, B = self.inputs
        names    = map(namefn, (alpha, detranspose(A), B))
        namemap  = dict(zip(varnames, names))
        other = {'TRANSA': trans(A), 'TRANSB': trans(B),
                 'LDA': LD(A), 'LDB': LD(B),
                 'M':str(B.shape[0]), 'N':str(B.shape[1]),
                 'fn': self.fnname(),
                 'SIDE': left_or_right(A, B, Q.triangular, assumptions),
                 'DIAG': diag(A, assumptions),
                 'UPLO': uplo(A, assumptions)}
        return merge(namemap, other)

class SM(BLAS):
    _inputs   = (alpha, S, A)
    _outputs  = (alpha*S.I*A,)
    view_map  = {0: 2}
    condition = True

class TRSM(SM):
    condition = Q.triangular(A)

class MV(BLAS):
    _inputs   = (alpha, A, a, beta, x)
    _outputs  = (alpha*A*a + beta*x,)
    view_map  = {0: 4}
    condition = True

class GEMV(MV):
    pass

class SYMV(MV):
    condition = Q.symmetric(A) | Q.symmetric(B)

class TRMV(MV):
    condition = Q.triangular(A) | Q.triangular(B)

class SV(BLAS):
    _inputs   = (S, x)
    _outputs  = (S.I*x,)
    view_map  = {0: 1}
    condition = True

class TRSV(SV):
    fortran_template = ("call %(fn)s('%(UPLO)s', '%(TRANS)s', '%(DIAG)s', "
                        "%(N)s, %(A)s, %(LDA)s, %(x)s, %(INCX)s)")
    condition = Q.triangular(A)
    def codemap(self, namefn, assumptions=True):
        varnames = 'A x'.split()
        A, x = self.inputs
        names    = map(namefn, (detranspose(A), x))
        namemap  = dict(zip(varnames, names))
        other = {'TRANS': trans(A),
                 'LDA': LD(A),
                 'N':str(A.shape[0]),
                 'fn': self.fnname(),
                 'DIAG': diag(A, assumptions),
                 'UPLO': uplo(A, assumptions),
                 'INCX': '1'}
        return merge(namemap, other)

# TODO: Make these classes
class Lof(Basic):    pass
class Uof(Basic):    pass
class LU(BLAS):
    _inputs   = (S,)
    _outputs  = (Lof(S), Uof(S))
    view_map  = {0: 0, 1: 0}
    condition = True

class Cholesky(LU):
    condition = Q.symmetric(S) & Q.positive_definite(S)

class CopyMatrix(InplaceComputation):
    _id = 0
    view_map = {}
    def __new__(cls, x):
        name = '_%d'%cls._id
        cls._id += 1
        cp = MatrixSymbol(name, x.rows, x.cols)
        return Computation.__new__((x,), (cp,))

def trans(A):
    if isinstance(A, Transpose):
        return 'T'
    else:
        return 'N'

def uplo(A, assumptions):
    if ask(Q.upper_triangular(A), assumptions):
        return 'U'
    if ask(Q.lower_triangular(A), assumptions):
        return 'L'

def LD(A):
    # TODO make sure we don't use transposed matrices in untransposable slots
    return str(detranspose(A).shape[0])

def left_or_right(A, B, predicate, assumptions):
    if ask(predicate(A), assumptions):
        return 'L'
    if ask(predicate(B), assumptions):
        return 'R'

def diag(A, assumptions):
    if ask(Q.unit_triangular(A), assumptions):
        return 'U'
    else:
        return 'N'

def detranspose(A):
    if isinstance(A, Transpose):
        return A.arg
    else:
        return A

def basic_names(x):
    if x in basic_names._cache:
        return basic_names._cache[x]
    if isinstance(x, (MatrixSymbol, Symbol)):
        result = x.name
    else:
        result = "_%d"%basic_names._id
        basic_names._id += 1
    basic_names._cache[x] = result
    return result
basic_names._cache = {}
basic_names._id = 1

class MatrixRoutine(CompositeComputation, MatrixComputation):
    """ A Composite MatrixComputation """
    def calls(self, namefn, assumptions=True):
        computations = map(self.inplace_fn(), self.toposort())
        return [call for c in computations
                     for call in c.calls(namefn, assumptions)]

    def types(self):
        return merge(*map(lambda x: x.types(), self.computations))

    @property
    def variables(self):
        return set([x for c in self.computations for x in c.variables])

    def replacements(self):
        return merge(*[c.replacements() for c in self.computations])
