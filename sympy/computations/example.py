from sympy.computations import Computation
from sympy import Min, Max, symbols, Basic, Add

x, y = symbols('x,y')
patterns = []

class inc(Computation):
    arg = property(lambda self: self.args[0])
    inputs = property(lambda self: (self.arg,))
    outputs= property(lambda self: (self.arg + 1,))


class double(Computation):
    arg = property(lambda self: self.args[0])
    inputs = property(lambda self: (self.arg,))
    outputs= property(lambda self: (2*self.arg,))


class add(Computation):
    inputs = property(lambda self: self.args)
    outputs= property(lambda self: (self.args[0] + self.args[1],))


class incdec(Computation):
    inputs  = property(lambda self: self.args)
    outputs = property(lambda self: (self.args[0] + 1, self.args[0] - 1))

class flipflop(Computation):
    inputs  = property(lambda self: self.args)
    outputs = property(lambda self: (Basic(self.args[0], self.args[1]),
                                     Basic(self.args[1], self.args[0])))

class minmax(Computation):
    inputs = property(lambda self: self.args)
    outputs= property(lambda self: (Min(self.args[0], self.args[1]),
                                    Max(self.args[0], self.args[1])))

    # multipattern = (set((Min(x, y), Max(x, y))), minmax(x, y), x, y)
patterns = ((x + 1, inc(x), x),
            (2*x, double(x), x),
            (x + y, add(x, y), x, y))