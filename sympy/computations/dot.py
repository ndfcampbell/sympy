
def declare_var(variable):
    if hasattr(variable, '_write_dot'):
        return variable._write_dot()
    return '"%s" [shape=ellipse]'%str(variable)

def declare_comp(comp):
    if hasattr(comp, '_write_dot'):
        return comp._write_dot()
    return '"%s" [shape=box, label="%s"]'%(str(comp), str(comp.__class__.__name__))

def call(comp):
    return '\n'.join(['"%s" -> "%s"' % tuple(map(str, edge)) for edge in
        comp.edges()])

template =\
'''
digraph{

%(flags)s

%(variables)s

%(computations)s

%(calls)s
}
'''

defaults={'orientation': 'TD'}

def gen_dot(computation, **kwargs):
    """ Generate DOT string for computation """
    flags = defaults.copy(); flags.update(kwargs)
    flags = '\n'.join(['%s=%s'%item for item in flags.items()])
    variables = '\n'.join(map(declare_var, computation.variables))
    computations = '\n'.join(map(declare_comp, computation.toposort()))
    calls = '\n'.join(map(call, computation.toposort()))
    return template%locals()

def writepdf(computation, filename, extension='pdf', **kwargs):
    import os
    with open(filename+'.dot', 'w') as f:
        f.write(gen_dot(computation, **kwargs))

    os.system('dot -T%s %s.dot -o %s.%s' % (
                    extension, filename, filename, extension))


def show(computation, filename='comp', extension='pdf', **kwargs):
    import os
    writepdf(computation, filename)
    os.system('evince %s.pdf' % filename)