import numpy as np
from typing import List

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

    # Set the order of variables so this fluxion can be callable
    def set_var_names(self, var_names: List[str]):
        """Associate a Fluxion with an ordered list of variable names."""
        self.var_names = var_names

    def bind_args(self, *args):
        """Bind arguments in a variable table"""
        # Create variables dictionary, var
        var_tbl = dict()
        for i, arg in enumerate(args):
            var_name = self.var_names[i]
            var_tbl[var_name] = arg
        return var_tbl


# *************************************************************************************************
class Unop(Fluxion):
    """Abstract class embodying a unary operation"""
    def __init__(self, f):
        self.f = f

class Const(Unop):
    def __init__(self, a):
        self.a = a

    def val(self, var=dict()):
        return self.a

    def diff(self, var=dict()):
        # The derivative of a constant is zero
        return 0

class Var(Unop):
    def __init__(self, nm):
        self.nm = nm

    def val(self, arg):
        """The diff method of a variable returns its value"""
        # If argument is a dictionary containing this variable name, 
        # assume it was passed in dictionary  (var_tbl) format and look it up by name
        if type(arg) == dict and self.nm in arg:
            return arg[self.nm]
        else:
            # Otherwise, assume arg was the value to be passed
            return arg

    def diff(self, var_tbl=dict()):
        """The diff method of a variable returns a 1 """
        return 1*(np.array(list(var_tbl)) == self.nm)

    def __call__(self, arg):
        """Make variables callable like functions"""
        return self.val(arg)


# *************************************************************************************************
class Binop(Fluxion):
    """Abstract class embodying a binary operation"""
    def __init__(self, f, g):
        f.is_node()
        g.is_node()
        self.f = f
        self.g = g

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

if __name__ == "__main__":
    # Examples

    # Create a variable, x
    x = Var('x')

    # f(x) = 5x
    f_x = 5 * x
    assert(f_x.val({'x':2}) == 10)
    assert(f_x.diff({'x':2}) == 5)

    # f(x) = 1 + (x * x)
    f_x = 1 + x * x
    assert(f_x.val({'x':2}) == 5)
    assert(f_x.diff({'x':3}) == 6)  

    # f(x) = (1 + x)/(x * x) 
    f_x = (1 + x) / (x * x)
    assert(f_x.val({'x':2}) == 0.75)
    assert(f_x.diff({'x':2}) == -0.5)

    # f(x) = (1 + 5x)/(x * x) 
    f_x = (1 + 5 * x) / (x * x)
    assert(f_x.val({'x':2}) == 2.75)
    assert(f_x.diff({'x':2}) == -1.5)

    print("Tests passed")
