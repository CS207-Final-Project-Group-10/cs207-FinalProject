# PyPI Name
name = "fluxions"

# The "public API" of the fluxions package
from .fluxion_node import Fluxion, Var, Unop, Const, Binop
from .fluxion_node import Addition, Subtraction, Multiplication, Division, Power
from .elementary_functions import DifferentiableUnopFunction, DifferentiableBinopFunction, DifferentiableFunctionFactory
# Trigonometric functions
from .elementary_functions import sin, cos, tan, arcsin, arccos, arctan, arctan2
# Miscellaneous trigonometric functions
from .elementary_functions import hypot, degrees, radians, deg2rad, rad2deg
# Hyperbolic functions
from .elementary_functions import sinh, cosh, tanh, arcsinh, arccosh, arctanh
# Exponents and logarithms
from .elementary_functions import exp, expm1, exp2, log, log10, log2, log1p, logaddexp, logaddexp2
# Miscellaneous functions
from .elementary_functions import sqrt, cbrt, square

# Jacobian
from .fluxion_jacobian import jacobian

# Define behavior of import * for sledgehammer fans
_modules = ['fluxion_node', 'elementary_functions']
_trig_functions_1 = ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2']
_trig_functions_2 = ['hypot', 'arctan2', 'degrees', 'radians', 'deg2rad', 'rad2deg']
_hyperbolic_functions = ['sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh']
_exponent_log_functions = ['exp', 'expm1', 'exp2', 'log', 'log10', 'log2', 'log1p', 'logaddexp', 'logaddexp2']
_miscellaneous_functions = ['sqrt', 'cbrt', 'square']
_jacobian = ['jacobian']
__all__ = _modules + _trig_functions_1 + _trig_functions_2 + _hyperbolic_functions + \
            _exponent_log_functions + _miscellaneous_functions + _jacobian
