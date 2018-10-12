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
        try:
            return Addition(self, other)
        except AttributeError:
            return Addition(self, Const(other))

    def __sub__(self, other):
        try:
            return Subtraction(self, other)
        except AttributeError:
            return Subtraction(self, Const(other))

    def __rsub__(self, other):
        try:
            return Subtraction(other, self)
        except AttributeError:
            return Subtraction(Const(other), self)

    def __mul__(self, other):
        try:
            return Multiplication(self, other)
        except AttributeError:
            return Multiplication(self, Const(other))

    def __rmul__(self, other):
        try:
            return Multiplication(self, other)
        except AttributeError:
            return Multiplication(self, Const(other))

    def __truediv__(self, other):
        try:
            return Division(self, other)
        except AttributeError:
            return Division(self, Const(other))

    def __rtruediv__(self, other):
        try:
            return Division(other, self)
        except AttributeError:
            return Division(Const(other), self)
        
class Binop(Node):
    def __init__(self, a, b):
        a.is_node()
        b.is_node()
        self.a = a
        self.b = b

class Unop(Node):
    def __init__(self, a):
        self.a = a

class Addition(Binop):
    def eval(self, vars={}):
        return self.a.eval(vars) + self.b.eval(vars)

    def diff(self, vars={}):
        return self.a.diff(vars) + self.b.diff(vars)

class Subtraction(Binop):
    def eval(self, vars={}):
        return self.a.eval(vars) - self.b.eval(vars)

    def diff(self, vars={}):
        return self.a.diff(vars) - self.b.diff(vars)

class Multiplication(Binop):
    def eval(self, vars={}):
        return self.a.eval(vars) * self.b.eval(vars)

    def diff(self, vars={}):
        return self.a.eval(vars) * self.b.diff(vars) + self.a.diff(vars) * self.b.eval(vars)

class Division(Binop):
    def eval(self, vars={}):
        return self.a.eval(vars) / self.b.eval(vars)

    def diff(self, vars={}):
        return (self.a.diff(vars) * self.b.eval(vars) - self.a.eval(vars) * self.b.diff(vars)) / (self.b.eval(vars) * self.b.eval(vars))

class Const(Unop):
    def eval(self, vars={}):
        return self.a

    def diff(self, vars={}):
        return 0

class Var(Unop):
    def eval(self, vars={}):
        return vars[self.a]

    def diff(self, vars={}):
        return 1

if __name__ == "__main__":
    # Examples
    # f(x) = 5x
    f_x = 5 * Var('x')
    assert(f_x.eval({'x':2}) == 10)
    assert(f_x.diff({'x':2}) == 5)

    # f(x) = 1 + (x * x)
    f_x = 1 + Var('x') * Var('x')
    assert(f_x.eval({'x':2}) == 5)
    assert(f_x.diff({'x':3}) == 6)  

    # f(x) = (1 + x)/(x * x) 
    f_x = (1 + Var('x')) / (Var('x') * Var('x'))
    assert(f_x.eval({'x':2}) == 0.75)
    assert(f_x.diff({'x':2}) == -0.5)

    # f(x) = (1 + 5x)/(x * x) 
    f_x = (1 + 5 * Var('x')) / (Var('x') * Var('x'))
    assert(f_x.eval({'x':2}) == 2.75)
    assert(f_x.diff({'x':2}) == -1.5)

    print("Tests passed")