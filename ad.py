class Node(object):
    def is_node(self):
        return True

    def __init__(self):
        self.is_node()

    def __add__(self, other):
        try:
            return Addition(self, other)
        except AttributeError:
            return Addition(self, Const(other))

    def __radd__(self, other):
        return self.__add__(self, other)

    def __mul__(self, other):
        try:
            return Multiplication(self, other)
        except AttributeError:
            return Multiplication(self, Const(other))

    def __rmul__(self, other):
        return self.__mul__(self, other)

class Multiplication(Node):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        a.is_node()
        b.is_node()

    def eval(self, vars={}):
        return self.a.eval(vars) * self.b.eval(vars)

    def diff(self, vars={}):
        return self.a.eval(vars) * self.b.diff(vars) + self.a.diff(vars) * self.b.eval(vars)

class Addition(Node):
    def __init__(self, a, b):
        a.is_node()
        b.is_node()
        self.a = a
        self.b = b

    def eval(self, vars={}):
        return self.a.eval(vars) + self.b.eval(vars)

    def diff(self, vars={}):
        return self.a.diff(vars) + self.b.diff(vars)

class Const(Node):
    def __init__(self, a):
        self.a = a

    def eval(self, vars={}):
        return self.a

    def diff(self, vars={}):
        return 0

class Var(Node):
    def __init__(self, var_name):
        self.var_name = var_name
    
    def eval(self, vars={}):
        return vars[self.var_name]
        
    def diff(self, vars={}):
        return 1