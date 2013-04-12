from sympy.computations.core import Computation
from sympy.matrices.expressions.fourier import DFT
from sympy import Symbol

class FFTWPlan(Computation):
  @property
  def outputs(self):
    return (Symbol('plan'),)

class FFTW(Computation):
  @property
  def inputs(self):
    return (self.args[0], Symbol('plan'))

  @property
  def outputs(self):
    return (DFT(self.inputs[0].shape[0]) * self.inputs[0], )

  fortran_template = (" ")

  @classmethod
  def codemap(cls, inputs, names, typecode, assumptions=True):
    pass
