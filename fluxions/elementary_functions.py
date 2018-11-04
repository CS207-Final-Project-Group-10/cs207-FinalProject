import numpy as np
from numpy import pi
from fluxions import Fluxion, Unop, Var

from typing import List, Callable, Union
value_type = Union[int, float, np.ndarray]
function_arg = Union[value_type, Fluxion]


# *************************************************************************************************
class differentiable_function(Unop):
    """Factory for analytically differentiable functions"""

    def __init__(self, func: Callable, deriv: Callable, func_name: str, var_names: Union[str, List[str]] = 'x'):
        """func is the value of the function, deriv is the derivative"""
        # Bind the function evaluation and derivative
        self.func = func
        self.deriv = deriv
        # Set function name and var names
        self.func_name = func_name
        self.set_var_names(var_names)

    def val(self, arg):        
        try:
            # If the argument was a fluxion, run the val() method on it, then evaluate the function
            return self.func(arg.val())
        except:
            # If the argument was a value type, evaluate the function
            return self.func(arg)

    def diff(self, arg):
        try:
            # If the argument was a fluxion, run the val() method on it, then evaluate the derivative
            # *then* apply the chain rule!
            return self.deriv(arg.val()) * arg.diff()
        except:
            # If the argument was a value type, evaluate the derivative
            return self.deriv(arg)

    def __repr__(self):
        return f'{self.func_name}'

    def __call__(self, arg):
        val = self.val(arg)
        diff = self.diff(arg)
        return (val, diff)
    

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
sin = differentiable_function(np.sin, np.cos, 'sin')

# The derivative of cos(x) is -sin(x)
def _minus_sin(x):
    """Return -sin(x); for the derivative of cos(x)"""
    return -np.sin(x)

cos = differentiable_function(np.cos, _minus_sin, 'cos')

# The derivative of tan(x) is sec^2(x)
def _sec2(x):
    """Return sec^2(x); for the derivative of tan(x)"""
    cx = np.cos(x)
    return 1.0 / (cx * cx)

tan = differentiable_function(np.tan, _sec2, 'tan')


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

arcsin = differentiable_function(np.arcsin, _deriv_arcsin, 'arcsin')

# The derivative of arccos(x) is -1 times the derivative of arcsin(x)
def _deriv_arccos(x):
    """The derivative of arccos(x)"""
    return -1.0 / np.sqrt(1.0 - x*x)

arccos = differentiable_function(np.arccos, _deriv_arccos, 'arccos')

# The derivative of arctan is 1 / (1+x^2)
def _deriv_arctan(x):
    """The derivative of arctan(x)"""
    return 1.0 / (1.0 + x*2)

arctan = differentiable_function(np.arctan, _deriv_arctan, 'arctan')


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

hypot = differentiable_function(np.hypot, _deriv_hypot, 'hypot', ['x', 'y'])

# atan2(y, x) = atan(y/x) with the angle chosen in the correct quadrant
# https://en.wikipedia.org/wiki/Atan2
def _deriv_arctan2(y, x):
    """Derivative of the arctan2 function"""
    r2 = x*x + y*y
    df_dy = x / r2
    df_dx = -y / r2
    return np.vstack([df_dy, df_dx].T)


arctan2 = differentiable_function(np.arctan2, _deriv_arctan2, 'arctan2', ['y', 'x'])

# radians(x) = k*x, where k is the number of degrees in one radian
# rad2deg(x) is an alias for degrees(x)
_k_rad2deg = np.degrees(1.0)
def _deriv_rad2deg(x):
    return _k_rad2deg

rad2deg = differentiable_function(np.rad2deg, _deriv_rad2deg, 'rad2deg', 'x') 
degrees = differentiable_function(np.degrees, _deriv_rad2deg, 'degrees', 'x') 

# radians(x) = k*x, where k is the number of radians in one degree
_k_deg2rad = np.radians(1.0)
def _deriv_deg2rad(x):
    return _k_deg2rad

deg2rad = differentiable_function(np.deg2rad, _deriv_deg2rad, 'deg2rad', 'x')
radians = differentiable_function(np.radians, _deriv_deg2rad, 'radians', 'x')


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
sinh = differentiable_function(np.sinh, np.cosh, 'sinh', 'x')

# The derivative of cosh(x) is sinh(x)
cosh = differentiable_function(np.cosh, np.sinh, 'cosh', 'x')

# The derivative of tanh(x) is 1 / cosh^2(x)
def _deriv_tanh(x):
    """Derivative of tanh(x)"""
    cx = np.cosh(x)
    return 1.0 / cx * cx

tanh = differentiable_function(np.tanh, _deriv_tanh, 'tanh', 'x')

# The derivative of arcsinh is 1 / sqrt(x^2+1)
def _deriv_arcsinh(x):
    """The derivative of arcsinh(x)"""
    return 1.0 / np.sqrt(x*x + 1)

arcsinh = differentiable_function(np.arcsinh, _deriv_arcsinh, 'arcsinh', 'x')

# The derivative of arccosh is 1 / sqrt(x^2-1); only exists for 1 < x
def _deriv_arccosh(x):
    """The derivative of arccosh(x)"""
    return 1.0 / np.sqrt(x*x - 1)

arccosh = differentiable_function(np.arccosh, _deriv_arccosh, 'arccosh', 'x')

# The derivative of arctanh is 1 / (1-x^2); only exists for -1 < x < 1
def _deriv_arctanh(x):
    """The derivative of arctanh(x)"""
    return 1.0 / (1.0 - x*x)

arctanh = differentiable_function(np.arctanh, _deriv_arctanh, 'arctanh', 'x')

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
exp = differentiable_function(np.exp, np.exp, 'exp', 'x')

# The derivative of f(x) = exp(x) - 1 is exp(x)
expm1 = differentiable_function(np.expm1, np.exp, 'exp', 'x')

# The derivative of exp2(x) = 2^x is log(2) * 2^x
_log2 = np.log(2.0)
def _deriv_exp2(x):
    """The derivative of exp2(x)"""
    return _log2 * np.exp2(x)

exp2 = differentiable_function(np.exp2, _deriv_exp2, 'exp2', 'x')

# The derivative of log(x) is 1/x
# Use this hand-rolled function because the numpy reciprocal function doesn't handle integers well
def _recip(x):
    """Return the reciprocal of x; to be used as the derivative of log(x)"""
    return 1.0 / x

log = differentiable_function(np.log, _recip, 'log', 'x')

# The derivative of log10(x) is (1 / log(10) * 1 / x
_log10 = np.log(10.0)
def _deriv_log10(x):
    """The derivative of log10(x)"""
    return 1.0 / (_log10 * x)

log10 = differentiable_function(np.log10, _deriv_log10, 'log10', 'x')


# The derivative of log2(x) is (1 / log(2) * 1 / x
def _deriv_log2(x):
    """The derivative of log2(x)"""
    return 1.0 / (_log2 * x)

log2 = differentiable_function(np.log2, _deriv_log2, 'log2', 'x')

def _deriv_logaddexp(x1, x2):
    """The derivative of f(x, y) = log(e^x + e^y)"""
    y1 = np.exp(x1)
    y2 = np.exp(x2)
    df_dx1 = y1 / (y1 + y2)
    df_dx2 = y2 / (y1 + y2)
    return np.vstack([df_dx1, df_dx2]).T

logaddexp = differentiable_function(np.logaddexp, _deriv_logaddexp, 'logaddexp', '[x1, x2]')


def _deriv_logaddexp2(x1, x2):
    """The derivative of f(x, y) = log2(2^x + 2^y)"""
    y1 = np.exp2(x1)
    y2 = np.exp2(x2)
    df_dx1 = y1 / (y1 + y2)
    df_dx2 = y2 / (y1 + y2)
    return np.vstack([df_dx1, df_dx2]).T

logaddexp2 = differentiable_function(np.logaddexp2, _deriv_logaddexp2, 'logaddexp2', '[x1, x2]')


# *************************************************************************************************
# Basic testing
# Create a variable theta with angles from 0 to 360 degrees, with values in radians
theta_val = np.linspace(0, 2*pi, 361)
theta = Var('theta', theta_val)

# Scalar version
y, dy_dx = sin(2)
assert y == np.sin(2)
assert dy_dx == np.cos(2)

# Vector version
y, dy_dx = sin(theta)
assert np.all(y == np.sin(theta_val))
assert np.all(dy_dx == np.cos(theta_val))

print('Testing of differentiable functions for f(x) = sin(x):   **** PASS ****')

