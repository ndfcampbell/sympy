from sympy import Basic

def make_getname():
    cache = {}
    seen = set(['', None])

    def getname(key, requested=None):
        """ Get the name associated to a key """
        if key in cache:
            return cache[key]

        name = requested
        if not name and hasattr(key, 'name'):
            name = key.name
        else:
            name = ''
        if name in seen:
            id = 2
            while(name + '_%d'%id in seen):
                id += 1
            name = name + '_%d'%id

        cache[key] = name
        seen.add(name)
        return name
    return getname


class Copy(Basic):
    arg = property(lambda self: self.args[0])
    tag = property(lambda self: self.args[1])

    name = property(lambda self: self.arg.name if hasattr(self.arg, 'name')
                                               else 'copy')

def inplace(x):
    try:
        return x.inplace
    except AttributeError:
        return {}
