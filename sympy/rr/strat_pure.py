# Generic strategies. No dependence on SymPy

def exhaust(rule):
    def exhaustive_rl(expr):
        new, old = rule(expr), expr
        while(new != old):
            new, old = rule(new), new
        return new
    return exhaustive_rl

def memoize(rule):
    cache = {}
    def memoized_rl(expr):
        if expr in cache:
            return cache[expr]
        else:
            result = rule(expr)
            cache[expr] = result
            return result
    return memoized_rl

def condition(cond, rule):
    def conditioned_rl(expr):
        if cond(expr): return rule(expr)
        else         : return      expr
    return conditioned_rl

def chain(*rules):
    def chain_rl(expr):
        for rule in rules:
            expr = rule(expr)
        return expr
    return chain_rl

def debug(rule):
    def debug_rl(expr):
        result = rule(expr)
        print "In: %s\nOut: %s\n"%(expr, result)
        return result
    return debug_rl