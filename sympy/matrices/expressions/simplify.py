# Gather and assert known relations
from logpy import Relation, facts
from simplifydata import known_relations, vars

reduces = Relation('reduces')
facts(reduces, *known_relations)

# Simplification code
from sympy.logpy import refine_one
from functools import partial
simplify_one = partial(refine_one, reduces=reduces, vars=vars)
