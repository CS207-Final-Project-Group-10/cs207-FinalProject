import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Union, Optional

# Type alias for a value_type; this is an integer, float, or a numpy array
scalar_type = Union[int, float]
value_type = Union[int, float, np.ndarray]
var_names_type = Union[str, List[str]]

# Tuples of types for use with isinstance
scalar_instance_types = (int, float)
value_instance_types = (int, float, np.ndarray)


# *************************************************************************************************
class FluxionInputType(Enum):
    """Different shapes of input that can be passed to a Fluxion in forward mode"""
    ARRAY_N = 1
    ARRAY_TxN = 2
    DICT = 3
    ARGS = 4
    KWARGS =5


# *************************************************************************************************
class Fluxion:
    """A Fluxion embodies a differentiable function"""

    def __init__(self, m: int=1, n: int=1, 
                 name: Optional[str] = None, var_names: Optional[var_names_type]=None) -> None:
        """
        Instantiate a fluxion.
        INPUTS:
        ======
        m:      space R^m that this function maps to
        n:      space R^n that this function maps from
                f is a mapping from R^m to R^n and has an mxn Jacobian 
        name:   the name of this fluxion
        var_names: the names of the input variables to this fuxion
        """
        # Set the dimensions of this fluxion
        self.m: int = m
        self.n: int = n
        # Initialize the number of samples to 0
        self.T: int = 0
        # Set the name
        if name is None:
            self.name = 'f'
        else:
            self.name = name
        # Set the variable names
        if var_names is None:
            # If no variable names were provided, use x or [x1, ..., xn] as placeholders
            if n == 0:
                self.set_var_name('x')
            else:
                self.set_var_names([f'x{i}' for i in range(1, n+1)])
        else:
            # Bind the provided variable names
            self.set_var_names(var_names)

    def is_node(self):
        """Used for implicit promotion of constants"""
        return True

    def shape(self) -> Tuple[int, int]:
        """The shape of this fluxion according to numpy standard"""
        return (self.m, self.n)

    # Set sizes
    def set_m(self, m: int) -> None:
        """Set the number of output dimensions"""
        self.m = m

    def set_n(self, n: int) -> None:
        """Set the number of input dimensions"""
        self.n = n

    def set_T(self, T: int) -> None:
        """Set the number of samples, T"""
        self.T = T

    # Set the order of variables so this fluxion can be callable
    def set_var_names(self, var_names: List[str]):
        """Update the ordered list of variable names, _var_names."""
        if isinstance(var_names, str):
            self.var_names = [var_names]
            self.n = len(var_names)
        elif isinstance(var_names, list):
            self.var_names = var_names
            self.n = len(var_names)
        else:
            raise TypeError('var_names must be a string or a list of strings')

    def set_var_name(self, var_name: str):
        """Set the name of a single variable."""
        self.set_var_names([var_name])

    def check_forward_mode_input(self, X: np.ndarray) -> int:
        """
        Check whether one forward mode input array is of valid shape
        Returns inferred value of T
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('Must be a numpy array')
        # Get the shape and tensor rank
        shape = X.shape
        tensor_rank = len(shape)
        # Only 1D and 2D arrays are supported
        if tensor_rank not in (1, 2):
            raise ValueError('Numpy array must be a 1D vector or 2D matrix')        
        # If the input was a 1D vector, its length must EITHER (1) be n, or (2) n == 1
        if tensor_rank == 1 and (shape[0] != self.n) and (self.n != 1):
            raise ValueError(f'Error: X has shape {X.shape}, incompatible with n = {self.n} on fluxion.')
        # Return the value of T in this situation
        if tensor_rank == 1 and shape[0] == self.n:
            return 1
        if tensor_rank == 1 and self.n == 1:
            return shape[0]
        # If the input was a 2D vector, it must be of shape Txn
        if tensor_rank == 2 and (shape[1] != self.n):
            raise ValueError(f'Error: X has shape {X.shape}, incompatible with n = {self.n} on fluxion.')
        return shape[0]            

    def check_forward_mode_inputs(self, X: np.ndarray, dX: np.ndarray) -> None:
        self.check_forward_mode_input(X)
        self.check_forward_mode_input(dX)

    def parse_var_tbl(self, var_tbl, default: Optional[float]=None):
        """Parse a table of variable bindings (dictionary with key = variable name, value = scalar or array."""
        n: int = self.n
        # print(f'processing dict...')
        # Find the first match of a variable to infer whether these are scalars or arrays
        T: int = 1
        for var_name in self.var_names:
            if var_name in var_tbl:
                # The bound value to this variable name
                val = var_tbl[var_name]
                # case 1: this is a scalar; T=1
                if isinstance(val, scalar_instance_types):
                    T = 1
                elif isinstance(val, np.ndarray):
                    T = self.check_forward_mode_input(val)
                else:
                    raise ValueError(f'val={val} in var_tbl; not a recognized value type.')
                # once the first variable is found, assume the rest share the same T
                break
        # The shape of X based on T and n
        if T > 1 and n > 1:
            shape = (T, n)
            X = np.zeros((T, n))            
        elif T > 1 and n == 1:
            shape = (T,)
        elif T == 1:
            shape = (n,)
        # Initialize X to zeros in the correct shape        
        X = np.zeros(shape)

        # Fill in variables by their slot
        def var_lookup(var_name: str):
            if default is None:
                return var_tbl[var_name]
            else:
                return var_tbl.get(var_name, default)

        for j, var_name in enumerate(self.var_names):
            # All the input values must be specified
            if shape == (T, n):
                X[:,j] = var_lookup(var_name)
            elif shape == (T,):
                X[:] = var_lookup(var_name)
            else:
                X[j] = var_lookup(var_name)
        return X        

    def parse_args_val(self, *args) -> np.ndarray:
        """
        Parse input arguments used in function evaluation.
        End result will be one array X of shape (n) or (T, n)
        Allowed input shapes are:
        (1) ARRAY_N:    array of size n
        (2) ARRAY_TxN:  arrays of size Txn
        (3) DICT:       dictionary mapping variable names to values
        (4) ARGS:       a list of n values; n variables in order
        (5) KWARGS:     a kwargs list (currently not supported)
        """
        # initialize the X variable that will be returned
        X: np.ndarray = np.ndarray(0)
        # Get the number of arguments and inputs
        argc: int = len(args)
        n: int = self.n
        # Check each type in turn
        # The most common case is two arguments were passed for the vars and seeds
        # They can both be numpy arrays or dicts
        # print(f'Fluxion.parse_args_val, argc={argc}, n={n}.')
        if argc == 0:
            return None
        if argc == 1:
            arg_vars = args[0]
            # Case one: one numpy array
            if isinstance(arg_vars, np.ndarray):
                X = arg_vars
            # Check whether we got a dict
            elif isinstance(arg_vars, dict):
                X = self.parse_var_tbl(arg_vars)
            # If we've filled in arrays X and dX at this point, check their sizes
            if X.size > 0:
                # Check that the two shapes are valid.
                self.check_forward_mode_input(X)
                # Return the array of argument values
                return X
        # If we reach here, we either got ARGS or KWARGS
        if argc == n:
            # Initialize X and dX to zeros in the correct shape
            X = np.array(args)
            # Return the two arrays
            # Check that the two shapes are valid.
            self.check_forward_mode_input(X)
            return X
        # KWARGS not yet supported
        msg = f'argc={argc}'
        for arg in args:
            msg += f'{arg}'
        raise ValueError(f'Unrecognized input type for Fluxion.parse_args.  Details: \n{msg}')


    def parse_args_forward_mode(self, *args) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse input arguments used in forward mode differentiation.
        End result will be two arrays X and dX, each of shape (n) or (T, n)
        Allowed input shapes are:
        (1) ARRAY_N:    two arrays of size n
        (2) ARRAY_TxN:  two arrays of size Txn
        (3) DICT:       two dictionaries, each mapping variable names to values
        (4) ARGS:       a list of 2n values; n variables followed by n seeds
        (5) KWARGS:     a kwargs list (currently not supported)
        """
        # Types for X and dX
        X: np.ndarray = np.ndarray(0)
        dX: np.ndarray = np.ndarray(0)
        # Get the number of arguments and inputs
        argc: int = len(args)
        n: int = self.n
        # Check each type in turn
        # The most common case is two arguments were passed for the vars and seeds
        # They can both be numpy arrays or dicts
        # print(f'Entering parse_args_forward_mode. argc={argc}, n={n}')
        # for arg in args:
        #    print(f'{arg}')
        if argc == 0:
            return None, None
        # In the special case that argc == 1 and  n == 1, process X and default dX =1
        if argc == 1 and n == 1:
            # The lone argument
            arg = args[0]
            if isinstance(arg, dict):
                X = np.array([arg[self.var_names[0]]])
            elif isinstance(arg, scalar_instance_types):
                X = np.array(args)
            dX = np.ones_like(X)
            return X, dX
        if argc == 2:
            # print(f'argc == 2')
            arg_vars = args[0]
            arg_seed = args[1]
            # Case one: two numpy arrays
            if isinstance(arg_vars, np.ndarray) and isinstance(arg_seed, np.ndarray):
                # print(f'Var.parse_forward_mode - processing numpy arrays...')
                X = arg_vars
                dX = arg_seed
            # Check whether we got a pair of dicts
            elif isinstance(arg_vars, dict) and isinstance(arg_seed, dict):
                # Don't fill in missing values of X
                X = self.parse_var_tbl(arg_vars)
                # Fill in missing values in dX with 0
                dX = self.parse_var_tbl(arg_seed, default=0.0)
                # print(f'X={X}')
                # print(f'dX={dX}')
            # If we've filled in arrays X and dX at this point, check their sizes
            if X.size > 0 and dX.size > 0:
                # Check that the two shapes are valid.
                self.check_forward_mode_input(X)
                self.check_forward_mode_input(dX)
                # Return the two arrays
                return (X, dX)
        # If we reach here, we either got ARGS or KWARGS
        if argc == 2 * n:
            # Initialize X and dX to zeros in the correct shape
            X = np.array(args[0:n])
            dX = np.array(args[n:2*n])
            # Return the two arrays
            # Check that the two shapes are valid.
            self.check_forward_mode_input(X)
            self.check_forward_mode_input(dX)
            return (X, dX)
        # KWARGS not yet supported
        msg = f'argc={argc}'
        for arg in args:
            msg += f'{arg}'
        raise ValueError(f'Unrecognized input type for Fluxion.parse_args.  Details:\n{msg}')

    def calc_T(self, X) -> int:
        """Calculate the number of samples, T, from the shape of X"""
        shape = X.shape
        tensor_rank: int = len(shape)
        if tensor_rank == 1:
            return 1
        if tensor_rank == 2:
            return shape[0]
        raise ValueError(f'Shape of X = {X.shape}. Must be either a 1D or 2D array.')

    # ***********************************************************************************************
    def val(self, *args):
        """Funcation evaluation; abstract base class"""
        raise NotImplementedError

    def forward_mode(self, *args):
        """Forward mode differentiation; abstract base class"""
        raise NotImplementedError

    def jacobian(self, *args):
        """The Jacobian."""
        val = self.val(*args)        
        return val

    def diff(self, *args):
        """Call forward_mode; discard value, only keep the derivative."""
        val, diff = self.forward_mode(*args)
        return diff

    def __call__(self, *args):
        """Make Fluxion object callable like functions"""
        # Run forward mode
        return self.forward_mode(*args)

    # ***********************************************************************************************
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

    def __pow__(self, p: int):
        return Power(self, p)
    
    def __neg__(self):
        return Multiplication(self, Const(-1))
            
       
# *************************************************************************************************
class Unop(Fluxion):
    """Abstract class embodying a unary operation"""
    def __init__(self, f: Fluxion):
        self.f = f        

class Const(Unop):
    """A function returning a constant; floats are implicitly promoted to instances of Const"""
    def __init__(self, a: scalar_type, name: Optional[str]=None):
        # Set default for name of a constant to 'c'
        if name is None:
            name = 'c'
        # Initialize the parent Fluxion class; for a constant m=1 and n=0 (it depends on no variables)  
        Fluxion.__init__(self, 1, 0, name, [])
        # Constants are only floats (ints can be promoted)
        if not isinstance(a, scalar_instance_types):
            raise ValueError(f'Error: {a} is of type {type(a)}, not a scalar type (int or float).')
        # Promote an integer to a float if necessary
        self.a: float = float(a)

    def val(self, *args):
        """Forward mode differentiation for a constant"""
        # Return the bound value regardless of the arguments passed
        return self.a

    def forward_mode(self, *args):
        """Forward mode differentiation for a constant"""
        # Return the bound value and a derivative of zero regardless of the argumetns passed
        val: float = self.a
        diff: float = 0.0
        return (val, diff)

    def __repr__(self):
        return f'Const({self.a})'

class Var(Unop):
    """Class embodying the concept of a variable that is an input to a function"""
    def __init__(self, var_name: str, initial_value: Optional[np.ndarray]=None):
        """Variables must be instantiated with a name; binding an initial value is optional"""
        if initial_value is not None:
            if not isinstance(initial_value, value_instance_types):
                raise ValueError('Initial value of a variable must be a value type.')
            X = np.array(initial_value)
            # T: int = self.calc_T(X)         
            if len(X.shape) == 2:
                n = X.shape[1]
                T = X.shape[0]
            else:
                n = 1
                T = 1
        else:
            n = 1
            T = 0
        # Initialize the parent Fluxion class
        self.m = 1
        self.n = n
        self.T = T
        self.name = var_name
        self.var_name = var_name
        self.var_names = [var_name]
        # The initial value of this variable
        self.X = initial_value

    def set_val(self, X: value_type):
        """Set the value of this variable"""
        if not isinstance(X, value_instance_types):
            raise ValueError(f'Error: {X} of type {type(X)} is not a valute type.  Must be int, float, or numpy array.')
        # If X was a scalar, bind the value
        if isinstance(X, scalar_instance_types):
            self.X = float(X)
            self.T = 1
        else:
            self.check_forward_mode_input(X)
            self.X = X
            self.T = self.calc_T(X)

    def val(self, *args):
        """Function evaluation for a variable"""
        # Parse arguments into a numpy array
        X: np.ndarray = self.parse_args_val(*args)
        if X is not None:
            return X
        return self.X

    def forward_mode(self, *args):
        """Forward mode differentiation for variables"""
        # print(f'Var.forward_mode')
        # print(f'argc={len(args)}')
        # for arg in args:
        #    print(f'{arg}')
        # Parse arguments into two numpy arrays
        X: np.ndarray
        dX: np.ndarray
        X, dX = self.parse_args_forward_mode(*args)
        # print(f'X={X}')
        # print(f'dX={dX}')
        # The value is X
        if X is not None:
            val = X
        else:
            val = self.X
        # The derivative is the seed dX
        if dX is not None:
            diff = dX
        else:
            diff = np.ones_like(val)
        # Return both arrays
        return (val, diff)

    def __repr__(self):
        return f'Var({self.var_name}, {self.X})'


class Power(Unop):
    """Raise a fluxion to the power p"""
    def __init__(self, f: Fluxion, p: float = 0.0):
        # Initialize the parent Fluxion class
        Fluxion.__init__(self, f.m, f.n, f'Power({f.name, p})', f.var_names)
        self.f = f
        self.p = p
    
    def val(self, *args):
        """Function evaluation for a power"""
        # Evaluate inner function self.f
        X: np.ndarray = self.f.val(*args)
        # The function value
        return X ** self.p

    def forward_mode(self, *args):
        """Forward mode differentiation for a constant"""
        # Evaluate inner function self.f
        X: np.ndarray
        dX: np.ndarray
        X, dX = self.f.forward_mode(*args)
        # Alias the power to p for legibility
        p: float = self.p
        # The function value
        val = X ** p
        # The derivative
        diff = p * X ** (p-1) * dX
        return (val, diff)

    def __repr__(self):
        return f'Power({self.f.name}, {self.p})'


# *************************************************************************************************
class Binop(Fluxion):
    """Abstract class embodying a binary operation"""
    def __init__(self, f: Fluxion, g: Fluxion):
        # Check that both f and g are fluxion nodes
        f.is_node()
        g.is_node()
        # Bind the input fluxions to members f and g of the binary operator
        self.f = f
        self.g = g       
        # Create the list of variable names
        var_names = []
        if not isinstance(f, Const):
            var_names += [f.name]
        if not isinstance(g, Const) and g.name not in var_names:
            var_names += [g.name]
        self.var_names = var_names
        # Check the shapes
        if f.m != g.m:
            raise ValueError(f'In {self.__repr__()}, ms f.m={f.m} and g.m={g.m} must match for binary operation.')
        # Bind the shapes
        if f.m != g.m:
            raise ValueError(f'm on f and m must match.  f.m={f.m} g.m={g.m};')
        self.m = f.m
        self.n = len(self.var_names)


class Addition(Binop):
    """Addition (sum) of two fluxions; h = f + g"""

    def __init__(self, f: Fluxion, g: Fluxion):
        # Initialize the parent Binop class
        Binop.__init__(self, f, g)
        # Create the name of this fluxion
        self.name = f'Addition({str(self.f)}, {str(self.g)})'

    def val(self, *args):
        """Function evaluation a sum"""
        # (f+g)(x) = f(x) + g(x)
        return self.f.val(*args) + self.g.val(*args)

    def forward_mode(self, *args):
        """Forward mode differentiation for a sum"""
        # (f+g)(x) = f(x) + g(x)
        f_val, f_diff = self.f.forward_mode(*args)
        g_val, g_diff = self.g.forward_mode(*args)
        # The function value and derivative is the sum of f and g
        val = f_val + g_val
        diff = f_diff + g_diff
        return val, diff

    def __repr__(self):
        return f'Addition({str(self.f)}, {str(self.g)})'


class Subtraction(Binop):
    """Subtraction (difference) of two fluxions; h = f - g"""

    def __init__(self, f: Fluxion, g: Fluxion):
        # Initialize the parent Binop class
        Binop.__init__(self, f, g)
        # Create the name of this fluxion
        self.name = f'Subtraction({str(self.f)}, {str(self.g)})'

    def val(self, *args):
        """Function evaluation a difference"""
        # (f+g)(x) = f(x) - g(x)
        return self.f.val(*args) - self.g.val(*args)

    def forward_mode(self, *args):
        """Forward mode differentiation for a difference"""
        # (f-g)(x) = f(x) - g(x)
        f_val, f_diff = self.f.forward_mode(*args)
        g_val, g_diff = self.g.forward_mode(*args)
        # print(f'f_val = {f_val}')
        # print(f'g_val = {g_val}')
        # The function value and derivative is just the difference
        val = f_val - g_val
        diff = f_diff - g_diff
        return val, diff

    def __repr__(self):
        return f'Subtraction({str(self.f)}, {str(self.g)})'


class Multiplication(Binop):
    """Multiplication (product) of two fluxions; h = f * g"""

    def __init__(self, f: Fluxion, g: Fluxion):
        # Initialize the parent Binop class
        Binop.__init__(self, f, g)
        # Create the name of this fluxion
        self.name = f'Multiplication({str(self.f)}, {str(self.g)})'

    def val(self, *args):
        """Function evaluation for a product"""
        # (f*g)(x) = f(x) * g(x)
        # print(f'Multiplication.val')
        # print(f'args={args}')
        # f_val = self.f.val(*args)
        # g_val = self.g.val(*args)
        # print(f'f_val={f_val}')
        # print(f'g_val={g_val}')
        return self.f.val(*args) * self.g.val(*args)

    def forward_mode(self, *args):
        """Forward mode differentiation for a product"""
        # Product Rule of Calculus
        # https://en.wikipedia.org/wiki/Product_rule
        # (f*g)'(x) = f'(x) * g(x) + f(x) * g'(x)
        f_val, f_diff = self.f.forward_mode(*args)
        g_val, g_diff = self.g.forward_mode(*args)
        # The function value is the product (elementwise)
        val = f_val * g_val
        diff = f_diff * g_val + f_val * g_diff
        return val, diff


    def __repr__(self):
        return f'Multiplication({str(self.f)}, {str(self.g)})'

class Division(Binop):
    """Division (quotient) of two fluxions; h = f * g"""

    def __init__(self, f: Fluxion, g: Fluxion):
        # Initialize the parent Binop class
        Binop.__init__(self, f, g)
        # Create the name of this fluxion
        self.name = f'Division({str(self.f)}, {str(self.g)})'

    def val(self, *args):
        """Function evaluation for a product"""
        # (f*g)(x) = f(x) * g(x)
        return self.f.val(*args) / self.g.val(*args)

    def forward_mode(self, *args):
        # Quotient Rule of calculus
        # https://en.wikipedia.org/wiki/Quotient_rule
        # f(x) = g(x) / h(x),
        # f'(x) = (g('x)h(x) - g(x)h'(x)) / h(x)^2
        f_val, f_diff = self.f.forward_mode(*args)
        g_val, g_diff = self.g.forward_mode(*args)
        val = f_val / g_val
        diff = (f_diff * g_val - f_val * g_diff) / (g_val * g_val)
        return val, diff

    def __repr__(self):
        return f'Division({str(self.f)}, {str(self.g)})'
