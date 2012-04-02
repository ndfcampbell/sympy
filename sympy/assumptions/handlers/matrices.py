"""
Handlers for predicates related to Matrices : Symmetric, Identity, etc..."""
from sympy.logic.boolalg import conjuncts
from sympy.assumptions import Q, ask
from sympy.assumptions.handlers import CommonHandler

class AskSymmetricHandler(CommonHandler):
    """
    Handler for Q.integer
    Test that an expression belongs to the field of integer numbers
    """

    @staticmethod
    def MatrixSymbol(expr, assumptions):
        if ask(Q.identity(expr), assumptions):
            return True
        if Q.symmetric(expr) in conjuncts(assumptions):
            return True

    @staticmethod
    def MatrixBase(expr, assumptions):
        return expr.is_symmetric()

    @staticmethod
    def MatAdd(expr, assumptions):
        return all(ask(Q.symmetric(arg), assumptions) for arg in expr.args)

    @staticmethod
    def MatMul(expr, assumptions):
        return all(x.equals(y.T)
                for x,y in zip(expr.args, reversed(expr.args)))

    @staticmethod
    def MatPow(expr, assumptions):
        return ask(Q.symmetric(expr.base), assumptions) or expr.exp == 0

    @staticmethod
    def BlockMatrix(expr, assumptions):
        if (expr.rowblocksizes == expr.colblocksizes and
            all(expr.blocks[i,j].equals(expr.blocks[j,i])
                for i in range(blockshape[0])
                for j in range(blockshape[1]))):
            return True
        # can do ImmutableMatrix call here

    @staticmethod
    def Transpose(expr, assumptions):
        return ask(Q.symmetric(expr.arg, assumptions))

    @staticmethod
    def ZeroMatrix(expr, assumptions):
        return True

    Inverse = Transpose

class AskIdentityHandler(CommonHandler):
    """
    Handler for Q.rational
    Test that an expression belongs to the field of rational numbers
    """

    @staticmethod
    def Identity(expr, assumptions):
        return True

    @staticmethod
    def MatMul(expr, assumptions):
        """
        Rational ** Integer      -> Rational
        Irrational ** Rational   -> Irrational
        Rational ** Irrational   -> ?
        """
        if len(expr.args)==2:
            a, b = expr.args
            if (a==b.T and
                ask(Q.orthogonal(a), assumptions) and
                ask(Q.orthogonal(b), assumptions)):
                return True

