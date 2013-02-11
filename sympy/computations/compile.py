from sympy.unify.rewrite import rewriterule
import itertools as it

def input_crunch(etc):
    """ Turn an Expr->Comp rule into a Comp->Comp rule by looking at inputs

    etc is an Expr to Comp a function :: Expr -> Comp
    Returns a transformation :: Comp -> Comp
    """
    return lambda c: (c + x for x in it.chain(*map(etc, c.inputs)))

def multi_input_rule(sources, target, *wilds):
    from itertools import chain
    from sympy import FiniteSet, Symbol

    other = Symbol('_foo')
    source = FiniteSet(*sources)
    source2 = FiniteSet(other, *sources)
    rule = rewriterule(source, target, wilds)
    rule2 = rewriterule(source2, target, tuple(wilds)+(other,))

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
    return lambda c: (c + x for x in it.chain(*map(etc, c.outputs)))

def multi_output_rule(sources, target, *wilds):
    from itertools import chain
    from sympy import FiniteSet, Symbol

    other = Symbol('_foo')
    source = FiniteSet(*sources)
    source2 = FiniteSet(other, *sources)
    rule = rewriterule(source, target, wilds)
    rule2 = rewriterule(source2, target, tuple(wilds) + (other,))

    def outputs_brl(comp):
        outputs = FiniteSet(*comp.outputs)
        for c in chain(rule(outputs), rule2(outputs)):
            yield comp + c
    return outputs_brl
