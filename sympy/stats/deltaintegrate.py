from sympy import *

delta = deltafunctions.DiracDelta

def deltaintegrate(I):
    L = I.function.as_ordered_factors()
    d = [x for x in L if isinstance(x, delta)][0]
    L = [x for x in L if x is not d]
    f = Mul(*L)
    g = d.args[0]

    grad = Matrix([g.diff(v) for v in I.variables])
    mag_grad = sqrt((grad.T * grad)[0,0])

    # Separate limits into first and the rest
    first, limits = I.limits[0], I.limits[1:]
    x = first[0]
    list_of_solutions = solve(g, x)

    if not limits:
        return Add(*[(f/mag_grad).subs(x,soln) for soln in list_of_solutions])

    # Multivariate case
    integrals = []

    xlower, xupper = first[1:] # bounds
    for soln in list_of_solutions:
        for limit in limits:
            y, ylower, yupper = limit
            implied_lower = solve(soln - xlower, y)
            implied_upper = solve(soln - xupper, y)
            if len(implied_lower)==1 and len(implied_upper)==1: # simple case
                break
        if not (len(implied_lower)==1 and len(implied_upper)==1):
            raise ValueError("Limits/Delta on integral are too complex")

        # This needs to be fixed
        ylow = Max(implied_lower[0], ylower)
        yup = Min(implied_upper[0], yupper)

        new_limits = [l for l in limits if l is not limit] + [(y, ylow, yup)]

        integrand = (f / mag_grad).subs(x, soln)
        integrals.append(Integral(integrand, *new_limits))

    return Add(*integrals)







    new_f = Add(*[(f/mag_grad).subs(x, soln) for soln in list_of_solutions])





