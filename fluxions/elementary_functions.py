import importlib
import numpy as np
# Handle import of classes in fluxion_node differently based on import as module or run from test
if importlib.util.find_spec("fluxions") is not None:
    from fluxions import Fluxion, Unop
else:
    from fluxion_node import Fluxion, Unop


# Types and type aliases
from typing import List, Callable, Union
value_type = Union[int, float, np.ndarray]
function_arg = Union[value_type, Fluxion]
value_types = (int, float, np.ndarray)


# *************************************************************************************************
class DifferentiableInnerFunction(Unop):

    def __init__(self, f, func: Callable, deriv: Callable, func_name: str, var_names: Union[str, List[str]] = 'x'):
        self.f = f
        # Bind the function evaluation and derivative
        self.func = func
        self.deriv = deriv
        # Set function name and var names
        self.func_name = func_name
        self.var_names = var_names

    def arg_val_diff(self, arg):
        """Compute the value and derivative of the argument to the inner function."""
        if arg is None:
            # If no argument was provided, attempt to run the val() and diff() methods on the fluxion self.f
            try:
                val = self.f.val()
                diff = self.f.diff()
            except:
                raise RuntimeError(f'Can only evaluate a differentiable function with no arguments when '
                                   f'the inner function has a val() method that returns a value!')
        elif isinstance(arg, dict):
            # If the argument was a dictionary, run the val() method on the stored fluxion
            val = self.f.val(arg)
            diff = self.f.diff(arg)
            # print(f'Evaluated argument = {arg_value}')
        elif hasattr(arg, 'val') and hasattr(arg, 'diff'):
            # If the argument has val and diff methods, treat it as a fluxion: evaluate its val and diff.
            val = arg.val()
            diff = arg.diff()
        elif isinstance(arg, value_types):
            # If the argument was a value type, evaluate the function
            val = arg
            diff = 1.0
        else:
            raise RuntimeError(f'Error: arg {arg} of type {type(arg)} not a supported type.'
                               f'Type must be a value type (int, float, np.ndarray), a dict, '
                               f'or a Fluxion with a val() method.')
        return val, diff

    def val(self, arg=None):
        # Evaluate the argument; don't need its derivative
        arg_value, _ = self.arg_val_diff(arg)
        return self.func(arg_value)

    def diff(self, arg=None):
        arg_value, arg_diff = self.arg_val_diff(arg)
        try:
            diff_array = self.deriv(arg_value) * arg_diff
        except (ValueError):
            diff_array = np.zeros(np.shape(arg_diff))
            for i, (f,g) in enumerate(zip(self.deriv(arg_value),arg_diff)):
                diff_array[i]=f*g
        return diff_array

    def __call__(self, *args):
        return super(Unop, self).__call__(*args)

    def __repr__(self):
        return f'{self.func_name}({self.f})'


class DifferentiableFunction(Unop):
    """Factory for analytically differentiable functions"""

    def __init__(self, func: Callable, deriv: Callable, func_name: str, var_names: Union[str, List[str]] = 'x'):
        """func is the value of the function, deriv is the derivative"""
        # Bind the function evaluation and derivative
        self.func = func
        self.deriv = deriv
        # Set function name and var names
        self.func_name = func_name
        self.set_var_names(var_names)

    def __repr__(self):
        return f'{self.func_name}'

    def __call__(self, f):
        return DifferentiableInnerFunction(f, self.func, self.deriv, self.func_name, self.var_names)


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
sin = DifferentiableFunction(np.sin, np.cos, 'sin')

# The derivative of cos(x) is -sin(x)
def _minus_sin(x):
    """Return -sin(x); for the derivative of cos(x)"""
    return -np.sin(x)

cos = DifferentiableFunction(np.cos, _minus_sin, 'cos')

# The derivative of tan(x) is sec^2(x)
def _sec2(x):
    """Return sec^2(x); for the derivative of tan(x)"""
    cx = np.cos(x)
    return 1.0 / (cx * cx)

tan = DifferentiableFunction(np.tan, _sec2, 'tan')


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

arcsin = DifferentiableFunction(np.arcsin, _deriv_arcsin, 'arcsin')

# The derivative of arccos(x) is -1 times the derivative of arcsin(x)
def _deriv_arccos(x):
    """The derivative of arccos(x)"""
    return -1.0 / np.sqrt(1.0 - x*x)

arccos = DifferentiableFunction(np.arccos, _deriv_arccos, 'arccos')

# The derivative of arctan is 1 / (1+x^2)
def _deriv_arctan(x):
    """The derivative of arctan(x)"""
    return 1.0 / (1.0 + x*2)

arctan = DifferentiableFunction(np.arctan, _deriv_arctan, 'arctan')


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

hypot = DifferentiableFunction(np.hypot, _deriv_hypot, 'hypot', ['x', 'y'])

# atan2(y, x) = atan(y/x) with the angle chosen in the correct quadrant
# https://en.wikipedia.org/wiki/Atan2
def _deriv_arctan2(y, x):
    """Derivative of the arctan2 function"""
    r2 = x*x + y*y
    df_dy = x / r2
    df_dx = -y / r2
    return np.vstack([df_dy, df_dx].T)


arctan2 = DifferentiableFunction(np.arctan2, _deriv_arctan2, 'arctan2', ['y', 'x'])

# radians(x) = k*x, where k is the number of degrees in one radian
# rad2deg(x) is an alias for degrees(x)
_k_rad2deg = np.degrees(1.0)
def _deriv_rad2deg(x):
    return _k_rad2deg

rad2deg = DifferentiableFunction(np.rad2deg, _deriv_rad2deg, 'rad2deg', 'x') 
degrees = DifferentiableFunction(np.degrees, _deriv_rad2deg, 'degrees', 'x') 

# radians(x) = k*x, where k is the number of radians in one degree
_k_deg2rad = np.radians(1.0)
def _deriv_deg2rad(x):
    return _k_deg2rad

deg2rad = DifferentiableFunction(np.deg2rad, _deriv_deg2rad, 'deg2rad', 'x')
radians = DifferentiableFunction(np.radians, _deriv_deg2rad, 'radians', 'x')


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
sinh = DifferentiableFunction(np.sinh, np.cosh, 'sinh', 'x')

# The derivative of cosh(x) is sinh(x)
cosh = DifferentiableFunction(np.cosh, np.sinh, 'cosh', 'x')

# The derivative of tanh(x) is 1 / cosh^2(x)
def _deriv_tanh(x):
    """Derivative of tanh(x)"""
    cx = np.cosh(x)
    return 1.0 / cx * cx

tanh = DifferentiableFunction(np.tanh, _deriv_tanh, 'tanh', 'x')

# The derivative of arcsinh is 1 / sqrt(x^2+1)
def _deriv_arcsinh(x):
    """The derivative of arcsinh(x)"""
    return 1.0 / np.sqrt(x*x + 1)

arcsinh = DifferentiableFunction(np.arcsinh, _deriv_arcsinh, 'arcsinh', 'x')

# The derivative of arccosh is 1 / sqrt(x^2-1); only exists for 1 < x
def _deriv_arccosh(x):
    """The derivative of arccosh(x)"""
    return 1.0 / np.sqrt(x*x - 1)

arccosh = DifferentiableFunction(np.arccosh, _deriv_arccosh, 'arccosh', 'x')

# The derivative of arctanh is 1 / (1-x^2); only exists for -1 < x < 1
def _deriv_arctanh(x):
    """The derivative of arctanh(x)"""
    return 1.0 / (1.0 - x*x)

arctanh = DifferentiableFunction(np.arctanh, _deriv_arctanh, 'arctanh', 'x')

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
exp = DifferentiableFunction(np.exp, np.exp, 'exp', 'x')

# The derivative of f(x) = exp(x) - 1 is exp(x)
expm1 = DifferentiableFunction(np.expm1, np.exp, 'exp', 'x')

# The derivative of exp2(x) = 2^x is log(2) * 2^x
_log2 = np.log(2.0)
def _deriv_exp2(x):
    """The derivative of exp2(x)"""
    return _log2 * np.exp2(x)

exp2 = DifferentiableFunction(np.exp2, _deriv_exp2, 'exp2', 'x')

# The derivative of log(x) is 1/x
# Use this hand-rolled function because the numpy reciprocal function doesn't handle integers well
def _recip(x):
    """Return the reciprocal of x; to be used as the derivative of log(x)"""
    return 1.0 / x

log = DifferentiableFunction(np.log, _recip, 'log', 'x')

# The derivative of log10(x) is (1 / log(10) * 1 / x
_log10 = np.log(10.0)
def _deriv_log10(x):
    """The derivative of log10(x)"""
    return 1.0 / (_log10 * x)

log10 = DifferentiableFunction(np.log10, _deriv_log10, 'log10', 'x')


# The derivative of log2(x) is (1 / log(2) * 1 / x
def _deriv_log2(x):
    """The derivative of log2(x)"""
    return 1.0 / (_log2 * x)

log2 = DifferentiableFunction(np.log2, _deriv_log2, 'log2', 'x')


# The derivative of log1p(x) = log(1 +x ) = 1 / (1+x)
def _deriv_log1p(x):
    """The derivative of log1p(x)"""
    return 1.0 / (1.0 + x)

log1p = DifferentiableFunction(np.log1p, _deriv_log1p, 'log1p', 'x')

def _deriv_logaddexp(x1, x2):
    """The derivative of f(x, y) = log(e^x + e^y)"""
    y1 = np.exp(x1)
    y2 = np.exp(x2)
    df_dx1 = y1 / (y1 + y2)
    df_dx2 = y2 / (y1 + y2)
    return np.vstack([df_dx1, df_dx2]).T

logaddexp = DifferentiableFunction(np.logaddexp, _deriv_logaddexp, 'logaddexp', '[x1, x2]')


def _deriv_logaddexp2(x1, x2):
    """The derivative of f(x, y) = log2(2^x + 2^y)"""
    y1 = np.exp2(x1)
    y2 = np.exp2(x2)
    df_dx1 = y1 / (y1 + y2)
    df_dx2 = y2 / (y1 + y2)
    return np.vstack([df_dx1, df_dx2]).T

logaddexp2 = DifferentiableFunction(np.logaddexp2, _deriv_logaddexp2, 'logaddexp2', '[x1, x2]')
