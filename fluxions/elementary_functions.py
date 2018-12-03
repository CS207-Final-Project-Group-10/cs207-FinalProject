import numpy as np
from importlib import util
# Handle import of classes in fluxion_node differently based on import as module or run from test
if util.find_spec("fluxions") is not None:
    from fluxions import Fluxion, Unop
else:
    from fluxion_node import Fluxion, Unop

# Types and type aliases
from typing import List, Callable, Union
value_type = Union[int, float, np.ndarray]
function_arg = Union[value_type, Fluxion]

# Tuples of types for use with isinstance
scalar_instance_types = (int, float)
value_instance_types = (int, float, np.ndarray)


# *************************************************************************************************
class FluxionResult(Fluxion):
    """Wrapper for the result of calling an elemenatry function on value objects"""
    def __init__(self, val, diff):
        self.m = 0
        self.n = 1
        self.var_names = {}
        self._val = val
        self._diff = diff

    def _forward_mode(self, *args):
        return (self._val, self._diff)

    def __repr__(self):
        return f'FluxionResult({self._val},{self._diff})'


# *************************************************************************************************
class DifferentiableUnopFunction(Unop):
    """A node on the calcuulation graph that is an analytically differentiable function."""
    def __init__(self, f: Fluxion, func: Callable, deriv: Callable, func_name: str):
        # Reference to the fluxion that is the input for this unary operation
        # (only operation done in Unop.__init__)
        # Initialize the parent Unop class
        Unop.__init__(self,f)

        # The function and its derivative
        self.func = func
        self.deriv = deriv
        # Name of this function
        self.func_name = func_name

    def _forward_mode(self, *args):
        """Forward mode differentiation for a constant"""
        # Evaluate inner function self.f
        X: np.ndarray
        dX: np.ndarray
        X, dX = self.f._forward_mode(*args)
        # The function value
        val = self.func(X)
        # The derivative
        diff = self.deriv(X) * dX
        return (val, diff)

    def __repr__(self):
        return f'{self.func_name}({str(self.f)})'

class DifferentiableBinopFunction(Unop):
    """A node on the calcuulation graph that is an analytically differentiable function."""
    def __init__(self, f: Fluxion, g: Fluxion, func: Callable, deriv: Callable, func_name: str):
        # Reference to the fluxion that is the input for this unary operation
        # (only operation done in Unop.__init__)
        # Initialize the parent Unop class
        Binop.__init__(self,f,g)

        # The function and its derivative
        self.func = func
        self.deriv = deriv
        # Name of this function
        self.func_name = func_name

    def _forward_mode(self, *args):
        """Forward mode differentiation for a constant"""
        # Evaluate inner function self.f
        X: np.ndarray
        dX: np.ndarray
        X, dX = self.f._forward_mode(*args)
        Y: np.ndarray
        dY: np.ndarray
        Y, dY = self.g._forward_mode(*args)
        # The function value
        val = self.func(X,Y)
        # The derivative (user is responsible for chain rule, etc.)
        diff = self.deriv(X,Y)
        return (val, diff)

    def __repr__(self):
        return f'{self.func_name}({str(self.f)})'
    
class DifferentiableFunctionFactory:
    """Factory for analytically differentiable functions"""

    def __init__(self, func: Callable, deriv: Callable, func_name: str):
        """func is the value of the function, deriv is the derivative"""
        # Bind the function evaluation and derivative
        self.func = func
        self.deriv = deriv
        # Set function name and var names
        self.func_name = func_name

    def __repr__(self):
        return f'DifferentiableFunctionFactory({self.func_name})'

    def __call__(self, *args):
        # Usually there will be only one argument
        argc: int = len(args)
        # Basic case: args is a single Fluxion instance
        if argc == 1 and isinstance(args[0], Fluxion):
            return DifferentiableUnopFunction(args[0], self.func, self.deriv, self.func_name)
        elif argc == 2 and isinstance(args[0], Fluxion) and isinstance(args[1], Fluxion):
            return DifferentiableBinopFunction(args[0], args[1], self.func, self.deriv, self.func_name)
        # Second case: some bag of arguments that the underling analytical function and its derivative can handle
        else:
            val = self.func(*args)
            diff = self.deriv(*args)
            return FluxionResult(val, diff)

# *************************************************************************************************
# List of mathematical functions in numpy
# https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.math.html


# *************************************************************************************************
# Trigonometric functions
# https://en.wikipedia.org/wiki/Differentiation_of_trigonometric_functions
# sin(x)
# cos(x)
# tan(x)

# The derivative of sin(x) is cos(x)
sin = DifferentiableFunctionFactory(np.sin, np.cos, 'sin')

# The derivative of cos(x) is -sin(x)
def _minus_sin(x):
    """Return -sin(x); for the derivative of cos(x)"""
    return -np.sin(x)

cos = DifferentiableFunctionFactory(np.cos, _minus_sin, 'cos')

# The derivative of tan(x) is sec^2(x)
def _sec2(x):
    """Return sec^2(x); for the derivative of tan(x)"""
    cx = np.cos(x)
    return 1.0 / (cx * cx)

tan = DifferentiableFunctionFactory(np.tan, _sec2, 'tan')


# *************************************************************************************************
# Inverse Trigonometric functions
# https://en.wikipedia.org/wiki/Differentiation_of_trigonometric_functions
# arcsin(x)
# arccos(x)
# arctan(x)

# The derivative of arcsin(x) is 1 / (sqrt(1-x^2)
def _deriv_arcsin(x):
    """The derivative of arcsin(x)"""
    return 1.0 / np.sqrt(1.0 - x*x)

arcsin = DifferentiableFunctionFactory(np.arcsin, _deriv_arcsin, 'arcsin')

# The derivative of arccos(x) is -1 times the derivative of arcsin(x)
def _deriv_arccos(x):
    """The derivative of arccos(x)"""
    return -1.0 / np.sqrt(1.0 - x*x)

arccos = DifferentiableFunctionFactory(np.arccos, _deriv_arccos, 'arccos')

# The derivative of arctan is 1 / (1+x^2)
def _deriv_arctan(x):
    """The derivative of arctan(x)"""
    return 1.0 / (1.0 + x*2)

arctan = DifferentiableFunctionFactory(np.arctan, _deriv_arctan, 'arctan')


# *************************************************************************************************
# "Miscellaneous" Trigonometric functions
# hypot
# arctan2
# degrees
# radians
# deg2rad
# rad2deg

# hypot(x, y) = (x^2 + y^2)^(1/2)
# let r = hypot(x, y)
# dhypot / dx = x / r
# dhypot / dy = y / r

def _deriv_hypot(x, y):
    """Derivative of numpy hypot function"""
    r = np.hypot(x, y)
    df_dx = x / r
    df_dy = y / r
    return np.vstack([df_dx, df_dy]).T

hypot = DifferentiableFunctionFactory(np.hypot, _deriv_hypot, 'hypot')

# atan2(y, x) = atan(y/x) with the angle chosen in the correct quadrant
# https://en.wikipedia.org/wiki/Atan2
def _deriv_arctan2(y, x):
    """Derivative of the arctan2 function"""
    r2 = x*x + y*y
    df_dy = x / r2
    df_dx = -y / r2
    return np.vstack([df_dy, df_dx].T)


arctan2 = DifferentiableFunctionFactory(np.arctan2, _deriv_arctan2, 'arctan2')

# radians(x) = k*x, where k is the number of degrees in one radian
# rad2deg(x) is an alias for degrees(x)
_k_rad2deg = np.degrees(1.0)
def _deriv_rad2deg(x):
    return _k_rad2deg

rad2deg = DifferentiableFunctionFactory(np.rad2deg, _deriv_rad2deg, 'rad2deg')
degrees = DifferentiableFunctionFactory(np.degrees, _deriv_rad2deg, 'degrees') 

# radians(x) = k*x, where k is the number of radians in one degree
_k_deg2rad = np.radians(1.0)
def _deriv_deg2rad(x):
    return _k_deg2rad

deg2rad = DifferentiableFunctionFactory(np.deg2rad, _deriv_deg2rad, 'deg2rad')
radians = DifferentiableFunctionFactory(np.radians, _deriv_deg2rad, 'radians')


# *************************************************************************************************
# Hyperbolic functions
# https://en.wikipedia.org/wiki/Hyperbolic_function
# sinh(x)
# cosh(x)
# tanh(x)
# arcsinh(x)
# arccosh(x)
# arctanh(x)

# The derivative of sinh(x) is cosh(x)
sinh = DifferentiableFunctionFactory(np.sinh, np.cosh, 'sinh')

# The derivative of cosh(x) is sinh(x)
cosh = DifferentiableFunctionFactory(np.cosh, np.sinh, 'cosh')

# The derivative of tanh(x) is 1 / cosh^2(x)
def _deriv_tanh(x):
    """Derivative of tanh(x)"""
    cx = np.cosh(x)
    return 1.0 / cx * cx

tanh = DifferentiableFunctionFactory(np.tanh, _deriv_tanh, 'tanh')

# The derivative of arcsinh is 1 / sqrt(x^2+1)
def _deriv_arcsinh(x):
    """The derivative of arcsinh(x)"""
    return 1.0 / np.sqrt(x*x + 1)

arcsinh = DifferentiableFunctionFactory(np.arcsinh, _deriv_arcsinh, 'arcsinh')

# The derivative of arccosh is 1 / sqrt(x^2-1); only exists for 1 < x
def _deriv_arccosh(x):
    """The derivative of arccosh(x)"""
    return 1.0 / np.sqrt(x*x - 1)

arccosh = DifferentiableFunctionFactory(np.arccosh, _deriv_arccosh, 'arccosh')

# The derivative of arctanh is 1 / (1-x^2); only exists for -1 < x < 1
def _deriv_arctanh(x):
    """The derivative of arctanh(x)"""
    return 1.0 / (1.0 - x*x)

arctanh = DifferentiableFunctionFactory(np.arctanh, _deriv_arctanh, 'arctanh')

# *************************************************************************************************
# Rounding functions are NOT differentiable! Skip these...


# *************************************************************************************************
# Sums, products and differences
# skip these for now - handled in the fluxions library instead


# *************************************************************************************************
# Exponents and logarithms
# exp(x)
# expm1(x)
# exp2(x)
# log(x)
# log10(x)
# log2(x)
# log1p(x)
# logaddexp(x1, x2)
# logaddexp2(x1, x2)

# The derivative of exp(x) is exp(x)
exp = DifferentiableFunctionFactory(np.exp, np.exp, 'exp')

# The derivative of f(x) = exp(x) - 1 is exp(x)
expm1 = DifferentiableFunctionFactory(np.expm1, np.exp, 'exp')

# The derivative of exp2(x) = 2^x is log(2) * 2^x
_log2 = np.log(2.0)
def _deriv_exp2(x):
    """The derivative of exp2(x)"""
    return _log2 * np.exp2(x)

exp2 = DifferentiableFunctionFactory(np.exp2, _deriv_exp2, 'exp2')

# The derivative of log(x) is 1/x
# Use this hand-rolled function because the numpy reciprocal function doesn't handle integers well
def _recip(x):
    """Return the reciprocal of x; to be used as the derivative of log(x)"""
    return 1.0 / x

log = DifferentiableFunctionFactory(np.log, _recip, 'log')

# The derivative of log10(x) is (1 / log(10) * 1 / x
_log10 = np.log(10.0)
def _deriv_log10(x):
    """The derivative of log10(x)"""
    return 1.0 / (_log10 * x)

log10 = DifferentiableFunctionFactory(np.log10, _deriv_log10, 'log10')


# The derivative of log2(x) is (1 / log(2) * 1 / x
def _deriv_log2(x):
    """The derivative of log2(x)"""
    return 1.0 / (_log2 * x)

log2 = DifferentiableFunctionFactory(np.log2, _deriv_log2, 'log2')


# The derivative of log1p(x) = log(1 +x ) = 1 / (1+x)
def _deriv_log1p(x):
    """The derivative of log1p(x)"""
    return 1.0 / (1.0 + x)

log1p = DifferentiableFunctionFactory(np.log1p, _deriv_log1p, 'log1p')

def _deriv_logaddexp(x1, x2):
    """The derivative of f(x, y) = log(e^x + e^y)"""
    y1 = np.exp(x1)
    y2 = np.exp(x2)
    df_dx1 = y1 / (y1 + y2)
    df_dx2 = y2 / (y1 + y2)
    return np.vstack([df_dx1, df_dx2]).T

logaddexp = DifferentiableFunctionFactory(np.logaddexp, _deriv_logaddexp, 'logaddexp')


def _deriv_logaddexp2(x1, x2):
    """The derivative of f(x, y) = log2(2^x + 2^y)"""
    y1 = np.exp2(x1)
    y2 = np.exp2(x2)
    df_dx1 = y1 / (y1 + y2)
    df_dx2 = y2 / (y1 + y2)
    return np.vstack([df_dx1, df_dx2]).T

logaddexp2 = DifferentiableFunctionFactory(np.logaddexp2, _deriv_logaddexp2, 'logaddexp2')
