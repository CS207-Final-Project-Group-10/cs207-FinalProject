import numpy as np
from typing import List, Tuple, Dict, Union
# Type alias for a value_type; this is an integer, float, or a numpy array
value_type = Union[int, float, np.ndarray]


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
    def __init__(self, f: Fluxion):
        self.f = f
        
    def set_var_name(self, nm: str):
        """Set the name of the single input variable to this unary operation."""
        self.set_var_names(nm)


class Const(Unop):
    """A function returning a constant; floats are implicitly promoted to instances of Const"""
    def __init__(self, a: Union[float, int]):
        # Promote an integer to a float if necessary
        self.a = float(a)

    def val(self, arg=None):
        return self.a

    def diff(self, arg=None):
        # The derivative of a constant is zero
        return 0.0

    def __repr__(self):
        return f'Const({self.a})'

class Var(Unop):
    """Class embodying the concept of a variable that is an input to a function"""
    def __init__(self, nm: str, initial_value=None):
        """Variables must be instantiated with a name; binding an initial value is optional"""
        # The name of this variable
        self.nm = nm
        # The initial value of this variable
        self.x = initial_value

    def set_val(self, x: value_type):
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
            return 1.0
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


class Power(Unop):
    """Raise a fluxion to the power p"""
    def __init__(self, f: Fluxion, p: float = 0.0):
        self.f = f
        self.p = p
    
    def val(self, arg=None, p: float = None):
        if p is None:
            p = self.p
        return self.f.val(arg)**p

    def diff(self, arg=None, p: float = None):
        if p is None:
            p = self.p
        return p * self.f.val(arg)**(p-1)
    
    def __repr__(self):
        return f'Power({self.f.nm}, {self.p})'

# *************************************************************************************************
class Binop(Fluxion):
    """Abstract class embodying a binary operation"""
    def __init__(self, f: Fluxion, g: Fluxion):
        f.is_node()
        g.is_node()
        self.f = f
        self.g = g

    def __call__(self, args=None):
        """Make binary operations callable like functions"""
        return self.val(args), self.diff(args)

class Addition(Binop):
    """Addition (sum) of two fluxions; h = f + g"""
    def val(self, args=None):
        # (f+g)(x) = f(x) + g(x)
        return self.f.val(args) + self.g.val(args)

    def diff(self, args=None):
        # (f+g)'(x) = f'(x) + g'(x)
        return self.f.diff(args) + self.g.diff(args)

    def __repr__(self):
        return f'Addition({str(self.f)}, {str(self.g)})'

class Subtraction(Binop):
    """Subtraction (difference) of two fluxions; h = f - g"""
    def val(self, args=None):
        # (f-g)(x) = f(x) - g(x)
        return self.f.val(args) - self.g.val(args)

    def diff(self, args=None):
        # (f-g)'(x) = f'(x) - g'(x)
        return self.f.diff(args) - self.g.diff(args)

    def __repr__(self):
        return f'Subtraction({str(self.f)}, {str(self.g)})'

class Multiplication(Binop):
    """Multiplication (product) of two fluxions; h = f * g"""
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
    """Division (quotient) of two fluxions; h = f * g"""
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


class Composition(Binop):
    """Composition of two functions; h = f(g) """
    def val(self, args=None):
        #(f(g))(x) = f( g(x))
        return self.f.val(self.g.val(args))
    
    def diff(self, args=None):
        # Chain rule of calculus
        # (f(g))'(x) = f'( g(x))
        return self.f.diff(self.g.val(args)) * self.g.diff(args)
