from sympy.unify.usympy import patternify
from sympy.unify.rewrite import rewriterule

def brulify(source, target, *wilds):
    pattern = patternify(source, *wilds)
    return rewriterule(pattern, target)

def input_crunch(etc):
    """ Turn an Expr->Comp rule into a Comp->Comp rule by looking at inputs

    etc is an Expr to Comp a function :: Expr -> Comp
    Returns a transformation :: Comp -> Comp
    """
    def input_brl(comp):
        for i in comp.inputs:
            for c in etc(i):
                yield comp + c
    return input_brl

def output_crunch(etc):
    """ Turn an Expr->Comp rule into a Comp->Comp rule by looking at outputs

    etc is an Expr to Comp a function :: Expr -> Comp
    Returns a transformation :: Comp -> Comp
    """
    def output_brl(comp):
        for o in comp.outputs:
            for c in etc(o):
                yield comp + c
    return output_brl
