from sympy.unify.usympy import patternify
from sympy.unify.rewrite import rewriterule

def brulify(source, target, *wilds):
    """ Turn a source/target/wild set into a branching rule """
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

def multi_input_rule(sources, target, *wilds):
    from itertools import chain
    from sympy import FiniteSet, Symbol

    other = Symbol('_foo')
    source = FiniteSet(*sources)
    source2 = FiniteSet(other, *sources)
    rule = rewriterule(patternify(source, *wilds), target)
    rule2 = rewriterule(patternify(source2, other, *wilds), target)

    def inputs_brl(comp):
        inputs = FiniteSet(*comp.inputs)
        for c in chain(rule(inputs), rule2(inputs)):
            yield comp + c
    return inputs_brl

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
