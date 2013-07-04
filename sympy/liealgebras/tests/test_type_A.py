from sympy.liealgebras.type_A import CartanType

def test_CartanType():
    ct = CartanType(5)
    assert ct.roots() == 30
