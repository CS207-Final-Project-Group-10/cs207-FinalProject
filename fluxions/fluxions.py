import numpy as np
from typing import List, Tuple, Dict, Union

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
    def set_var_names(self, var_names: Union[str, List[str]]):
        """Associate a Fluxion with an ordered list of variable names."""
        if isinstance(var_names, str):
            self.var_names = [var_names]
        elif isinstance(var_names, list):
            self.var_names = var_names
        else:
            raise TypeError('var_names must be a string or a list of strings')

    def bind_args(self, *args):
        """Bind arguments in a variable table"""
        # Create variables dictionary, var
        var_tbl = dict()
        for i, arg in enumerate(args):
            var_name = self.var_names[i]
            var_tbl[var_name] = arg
        return var_tbl

    def __call__(self, *args):
        """Make Fluxion object callable like a function"""
        # print(f'In Fluxion.__call__()')
        # print(f'args={args}')
        # Bind arguments into a variable table
        var_tbl = self.bind_args(*args)
        # print(f'Variable Table: {var_tbl}')
        return self.val(var_tbl), self.diff(var_tbl)
        

# *************************************************************************************************
class Unop(Fluxion):
    """Abstract class embodying a unary operation"""
    def __init__(self, f):
        self.f = f

class Const(Unop):
    def __init__(self, a):
        self.a = a

    def val(self, arg=None):
        return self.a

    def diff(self, arg=None):
        # The derivative of a constant is zero
        return 0.0

    def __repr__(self):
        return f'Const({self.a})'

class Var(Unop):
    def __init__(self, nm, initial_value=None):
        # The name of this variable
        self.nm = nm
        # The initial value of this variable
        self.x = initial_value

    def set_val(self, x):
        """Set the value of this variable"""
        self.x = x

    def val(self, arg=None):
        """The diff method of a variable returns its value"""
        # If no argument is passed, used the stored value
        if arg is None:
            return self.x
        # If argument is a dictionary containing this variable name, 
        # assume it was passed in dictionary  (var_tbl) format and look it up by name
        if isinstance(arg, dict) and self.nm in arg:
            return arg[self.nm]
        # Otherwise, assume arg was the value to be passed
        return arg

    def diff(self, arg=None):
        """The diff method of a variable returns a 1 """
        # If no argument is passed, the derivative of f(x) = x is 1
        if arg is None:
            return 1
        #if the arg was a scalar, it will not have a length
        try:
            l = len(arg[self.nm])
        except(TypeError):
            l = 1
        # If argument was a dictionary, look up to set 1s where the variable name is matched
        if isinstance(arg, dict):
            # If the variable table has length 1, return a scalar rather than an array of length 1
            if len(arg) == 1:
                return np.asarray([1.0]*l) if self.nm in arg else np.asarray([1.0]*l)
            else:
                return np.asarray([1.0*(np.array(list(arg)) == self.nm)]*l)
        # Otherwise, assume arg was a scalar; again the derivative is 1
        return np.asarray([1.0]*l)

    def __call__(self, arg=None):
        """Make variables callable like functions"""
        return self.val(arg), self.diff(arg)

    def __repr__(self):
        return f'Var({self.nm}, {self.x})'

# *************************************************************************************************
class Binop(Fluxion):
    """Abstract class embodying a binary operation"""
    def __init__(self, f, g):
        f.is_node()
        g.is_node()
        self.f = f
        self.g = g

    def __call__(self, args=None):
        """Make binary operations callable like functions"""
        return self.val(args), self.diff(args)

class Addition(Binop):
    def val(self, args=None):
        # (f+g)(x) = f(x) + g(x)
        return self.f.val(args) + self.g.val(args)

    def diff(self, args=None):
        # (f+g)'(x) = f'(x) + g'(x)
        return self.f.diff(args) + self.g.diff(args)

    def __repr__(self):
        return f'Addition({str(self.f)}, {str(self.g)})'

class Subtraction(Binop):
    def val(self, args=None):
        # (f-g)(x) = f(x) - g(x)
        return self.f.val(args) - self.g.val(args)

    def diff(self, args=None):
        # (f-g)'(x) = f'(x) - g'(x)
        return self.f.diff(args) - self.g.diff(args)

    def __repr__(self):
        return f'Subtraction({str(self.f)}, {str(self.g)})'

class Multiplication(Binop):
    def val(self, args=None):
        # (f*g)(x) = f(x) * g(x)
        return self.f.val(args) * self.g.val(args)

    def diff(self, args=None):
        # Product Rule of calculus
        # https://en.wikipedia.org/wiki/Product_rule#Examples
        # (f*g)'(x) = f'(x) + g(x) + f(x)*g'(x)
        #return self.f.val(args) * self.g.diff(args) + self.f.diff(args) * self.g.val(args)
        fval = self.f.val(args)
        gval = self.g.val(args)
        fdiff = self.f.diff(args)
        gdiff = self.g.diff(args)
        if np.linalg.norm(fval) == 0 or np.linalg.norm(gdiff) == 0:
            fval = 0
            gdiff = 0
        if np.linalg.norm(gval) == 0 or np.linalg.norm(fdiff) == 0:
            fdiff = 0
            gval = 0
        try:
            return fval * gdiff + fdiff * gval
        except (ValueError):
            leftsum = np.zeros(np.shape(gdiff))
            for i, (f,g) in enumerate(zip(fval,gdiff)):
                leftsum[i]=f*g
            rightsum = np.zeros(np.shape(fdiff))
            for i, (f,g) in enumerate(zip(fdiff,gval)):
                rightsum[i]=f*g
            return leftsum + rightsum

    def __repr__(self):
        return f'Multiplication({str(self.f)}, {str(self.g)})'

class Division(Binop):
    def val(self, args=None):
        #(f/g)(x) = f(x) / g(x)
        return self.f.val(args) / self.g.val(args)

    def diff(self, args=None):
        # Quotient Rule of calculus
        # https://en.wikipedia.org/wiki/Quotient_rule
        # f(x) = g(x) / h(x),
        # f'(x) = (g('x)h(x) - g(x)h'(x)) / h(x)^2
        fval = self.f.val(args)
        gval = self.g.val(args)
        fdiff = self.f.diff(args)
        gdiff = self.g.diff(args)
        if np.linalg.norm(fval) == 0 or np.linalg.norm(gdiff) == 0:
            fval = 0
            gdiff = 0
        if np.linalg.norm(gval) == 0 or np.linalg.norm(fdiff) == 0:
            fdiff = 0
            gval = 0
        try:
            return (fdiff * gval - fval * gdiff) / \
                    (gval * gval)
        except (ValueError):
            leftsum = np.zeros(np.shape(gdiff))
            for i, (f,g) in enumerate(zip(fdiff,gval)):
                leftsum[i]=f*g
            rightsum = np.zeros(np.shape(fdiff))
            for i, (f,g) in enumerate(zip(fval,gdiff)):
                rightsum[i]=f*g
            numerator = leftsum - rightsum
            denominator = np.power(gval,2)
            quotient = np.zeros(np.shape(numerator))
            for i, (n,d) in enumerate(zip(numerator,denominator)):
                quotient[i]=n/d
            return quotient

    def __repr__(self):
        return f'Division({str(self.f)}, {str(self.g)})'

if __name__ == "__main__":
    # Examples

    # Create a variable, x
    x = Var('x', 1.0)

    # f(x) = 5x
    f1 = 5 * x
    f1.set_var_names('x')
    # Evaluate f1(x) at the bound value of x
    assert(f1() == (5.0, 5.0))
    # Evaluate f1(x) using function calling syntax
    assert(f1(2) == (10.0, 5.0))
    # Evaluate f1(x) using dictionary binding syntax
    assert(f1.val({'x':2}) == 10)
    assert(f1.diff({'x':2}) == 5)
    
    # f(x) = 1 + (x * x)
    f2 = 1 + x * x
    f2.set_var_names('x')
    assert(f2(4.0) == (17.0, 8.0))
    assert(f2.val({'x':2}) == 5)
    assert(f2.diff({'x':3}) == 6)  

    # f(x) = (1 + x)/(x * x) 
    f3 = (1 + x) / (x * x)
    f3.set_var_names('x')
    assert(f3.val({'x':2}) == 0.75)
    assert(f3.diff({'x':2}) == -0.5)

    # f(x) = (1 + 5x)/(x * x) 
    f4 = (1 + 5 * x) / (x * x)
    f4.set_var_names('x')
    assert(f4.val({'x':2}) == 2.75)
    assert(f4.diff({'x':2}) == -1.5)

    print("Tests passed")
