import numpy as np

class Fluxion:
    """A Fluxion embodies a differentiable function"""

    def __init__(self):
        self.is_node()

    def is_node(self):
        """Used for implicit promotion of constants"""
        return True

    # Overload operators: +, -, *, /
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
        
class Binop(Fluxion):
    """Abstract class embodying a binary operation"""
    def __init__(self, f, g):
        f.is_node()
        g.is_node()
        self.f = f
        self.g = g

class Unop(Fluxion):
    """Abstract class embodying a unary operation"""
    def __init__(self, f):
        self.f = f

class Addition(Binop):
    def val(self, var=dict()):
        # (f+g)(x) = f(x) + g(x)
        return self.f.val(var) + self.g.val(var)

    def diff(self, var=dict()):
        # (f+g)'(x) = f'(x) + g'(x)
        return self.f.diff(var) + self.g.diff(var)

class Subtraction(Binop):
    def val(self, var=dict()):
        # (f-g)(x) = f(x) - g(x)
        return self.f.val(var) - self.g.val(var)

    def diff(self, var=dict()):
        # (f-g)'(x) = f'(x) - g'(x)
        return self.f.diff(var) - self.g.diff(var)

class Multiplication(Binop):
    def val(self, var=dict()):
        # (f*g)(x) = f(x) * g(x)
        return self.f.val(var) * self.g.val(var)

    def diff(self, var=dict()):
        # Product Rule of calculus
        # https://en.wikipedia.org/wiki/Product_rule#Examples
        # (f*g)'(x) = f'(x) + g(x) + f(x)*g'(x)
        return self.f.val(var) * self.g.diff(var) + self.f.diff(var) * self.g.val(var)

class Division(Binop):
    def val(self, var=dict()):
        #(f/g)(x) = f(x) / g(x)
        return self.f.val(var) / self.g.val(var)

    def diff(self, var=dict()):
        # Quotient Rule of calculus
        # https://en.wikipedia.org/wiki/Quotient_rule
        # f(x) = g(x) / h(x),
        # f'(x) = (g('x)h(x) - g(x)h'(x)) / h(x)^2
        return (self.f.diff(var) * self.g.val(var) - self.f.val(var) * self.g.diff(var)) / \
                (self.g.val(var) * self.g.val(var))

class Const(Unop):
    def val(self, var=dict()):
        return self.f

    def diff(self, var=dict()):
        # The derivative of a constant is zero
        return 0

class Var(Unop):
    def val(self, var=dict()):
        """The val method of a variable returns its value"""
        return var[self.f]

    def diff(self, var=dict()):
        """The diff method of a variable returns its value"""
        return 1*(np.array(list(var)) == self.f)

if __name__ == "__main__":
    # Examples
    # f(x) = 5x
    f_x = 5 * Var('x')
    assert(f_x.val({'x':2}) == 10)
    assert(f_x.diff({'x':2}) == 5)

    # f(x) = 1 + (x * x)
    f_x = 1 + Var('x') * Var('x')
    assert(f_x.val({'x':2}) == 5)
    assert(f_x.diff({'x':3}) == 6)  

    # f(x) = (1 + x)/(x * x) 
    f_x = (1 + Var('x')) / (Var('x') * Var('x'))
    assert(f_x.val({'x':2}) == 0.75)
    assert(f_x.diff({'x':2}) == -0.5)

    # f(x) = (1 + 5x)/(x * x) 
    f_x = (1 + 5 * Var('x')) / (Var('x') * Var('x'))
    assert(f_x.val({'x':2}) == 2.75)
    assert(f_x.diff({'x':2}) == -1.5)

    print("Tests passed")