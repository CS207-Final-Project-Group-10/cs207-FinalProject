import numpy as np
from numpy import pi
import sys
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
# sinh(x)
# cosh(x)
# tanh(x)
# arcsinh(x)
# arccosh(x)
# arctanh(x)


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

def _recip(x):
    """Return the reciprocal of x; to be used as the derivative of log(x)"""
    return 1.0 / x

# The derivative of log(x) is 1/x
log = differentiable_function(np.log, _recip, 'log', 'x')




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

