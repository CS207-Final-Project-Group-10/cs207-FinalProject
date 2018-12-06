import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Union, Optional

# Type alias for a value_type; this is an integer, float, or a numpy array
scalar_type = Union[int, float]
value_type = Union[int, float, np.ndarray]
var_names_type = dict

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

    def __init__(self) -> None:
        """
        Instantiate a fluxion.
        INPUTS:
        ======
        m:      space R^m that this function maps from
        n:      space R^n that this function maps to
                f is a mapping from R^m to R^n and has an mxn Jacobian 
        name:   the name of this fluxion
        """
        # Set the dimensions of this fluxion to 0
        self.m: int = 0
        self.n: int = 0
        # Initialize the number of samples to 0
        self.T: int = 0
        # Initialize the var names
        self.var_names = {}

    def shape(self) -> Tuple[int, int]:
        """The shape of this fluxion according to numpy standard"""
        return (self.m, self.n)

    # ***********************************************************************************************
    def val(self, *args):
        """Funcation evaluation; abstract base class"""
        arg_dicts = self._parse_args_forward_mode(*args)
        val, diff = self._forward_mode(*arg_dicts)
        return val

    def diff(self, *args):
        """Call forward_mode; discard value, only keep the derivative."""
        arg_dicts = self._parse_args_forward_mode(*args)
        val, diff = self._forward_mode(*arg_dicts)
        return diff

    def __call__(self, *args):
        """Make Fluxion object callable like functions"""
        # Run forward mode
        arg_dicts = self._parse_args_forward_mode(*args)
        val, diff = self._forward_mode(*arg_dicts)
        return (val, diff)

    def _forward_mode(self, *args):
        """Forward mode differentiation; abstract base class"""
        raise NotImplementedError

    # ***********************************************************************************************
    # The parsing function takes a variety of human readable input types and transforms them into our
    # an internal representation consistent among all fluxions objects. In principle, this is modular,
    # so we could pick any types that we wanted. For this implementation, we chose to pass all
    # arguments and dicts as we work our way down the computational graph. On the way back up, results 
    # are passed as numpy arrays.

    def _parse_args_forward_mode(self, *args) -> Tuple[dict, dict]:
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
        # Types for arg_vars and arg_seed
        arg_vars: np.ndarray = {}
        arg_seed: np.ndarray = {}
        # Get the number of arguments and inputs
        argc: int = len(args)
        m: int = self.m
        # Check each type in turn
        # The most common case is two arguments were passed for the vars and seeds
        # They can both be numpy arrays or dicts
        if argc == 0:
            return (None, None)
        if argc == 1:
            # The lone argument
            arg = args[0]
            # Case zero: None
            if arg is None:
                return (None, None)
            # Case one: a dict
            elif isinstance(arg, dict):
                arg_vars = self._parse_var_dict(arg)
            # Case two: a scalar
            elif isinstance(arg, scalar_instance_types):
                arg_vars = self._parse_var_scalar(arg)
            # Case three: a numpy array
            elif isinstance(arg, np.ndarray):
                arg_vars = self._parse_var_array(arg)
            T_vars = self._check_forward_mode_input_dict(arg_vars)
            self.T = T_vars
            return (arg_vars, self._default_seed(arg_vars))
        if argc == 2:
            # Case one: a pair of dicts
            if isinstance(args[0], dict) and isinstance(args[1], dict):
                arg_vars = self._parse_var_dict(args[0])
                arg_seed = self._parse_seed_dict(args[1])
            # Case two: two scalars
            elif isinstance(args[0], scalar_instance_types) and isinstance(args[1], scalar_instance_types):
                arg_vars = self._parse_var_scalar(args[0])
                arg_seed = self._parse_var_scalar(args[1])
            # Case three: two numpy arrays
            elif isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
                arg_vars = self._parse_var_array(args[0])
                arg_seed = self._parse_var_array(args[1])
            else:
                raise ValueError(f'Input types must either be dict, scalar, or np.ndarray for Fluxion.parse_args.')
            T_vars = self._check_forward_mode_input_dict(arg_vars)
            T_seed = self._check_forward_mode_input_dict(arg_seed)
            self.T = T_vars
            if T_seed in (1,T_vars):
                return (arg_vars, arg_seed)
            else:
                raise ValueError(f'Bound variables in {args[0]} inconsistent with bound variables in {args[1]}')

        # If we reach here, we either got ARGS or KWARGS
        if argc == 2 * m:
            # Initialize X and dX in the correct shape
            X = np.array(args[0:m],dtype=np.float64)
            dX = np.array(args[m:2*m],dtype=np.float64)
            # Reevaluate the two arrays
            return self._parse_args_forward_mode(X, dX)
        # KWARGS not yet supported
        msg = f'argc={argc}'
        for arg in args:
            msg += f'{arg}'
        raise ValueError(f'Unrecognized input type for Fluxion.parse_args.  Details:\n{msg}')

    def _default_seed(self, var_tbl: dict) -> dict:
        """
        Returns inferred dict of variable: val = 1 pairs
        """
        var_names = {}
        #for v in self.var_names:
        #    var_names[v]=1
        for v in var_tbl:
            var_names[v]=1
        return var_names

    def _parse_var_dict(self, var_tbl: dict) -> dict:
        """
        Create an extended list of variable names including any pre-bound ones
        Throws an error if a variable is unbound
        Returns inferred dict of variable: val pairs
        """
        var_names = var_tbl.copy()
        # check that all of the variables are bound
        for v in self.var_names:
            if v not in var_tbl:
                if self.var_names[v] is None:
                    raise(KeyError("variable " + v + " is unbound"))
                else:
                    var_names[v] = self.var_names[v]
        # squeeze the variables
        return self._squeeze(var_names)

    def _parse_seed_dict(self, var_tbl: dict) -> dict:
        """
        Create an extended list of variable names including any pre-bound ones
        Throws an error if a variable is unbound
        Returns inferred dict of variable: val pairs
        """
        var_names = var_tbl.copy()
        # check that all of the variables are bound
        for v in self.var_names:
            if v not in var_tbl:
                raise(KeyError("variable " + v + " is unbound in seed table"))
        # squeeze the variables
        return self._squeeze(var_names)

    def _squeeze(self, var_tbl: dict) -> dict:
        """
        Makes sure no extra dimensions are floating around in the input arrays
        Returns inferred dict of variable: val pairs
        """
        var_names = var_tbl.copy()
        # squeeze any numpy arrays
        for v in var_tbl:
            val = var_tbl[v]
            if isinstance(val, np.ndarray):
                var_names[v] = val.squeeze()
        return var_names
    
    def _parse_var_scalar(self, X: scalar_type) -> dict:
        """
        Unpack the numpy array and bind each column to one of the variables in self.var_names
        Returns inferred dict of variable: val pairs
        """
        arg_vars = {}
        #there is only one var name, so this will run once and no sorting is required
        for var_name in self.var_names:
            arg_vars[var_name] = X
        return arg_vars

    def _parse_var_array(self, X: np.ndarray) -> dict:
        """
        Unpack the numpy array and bind each column to one of the variables in self.var_names
        Returns inferred dict of variable: val pairs
        """
        arg_vars = {}
        # Get the shape and tensor rank
        shape = X.shape
        tensor_rank = len(shape)
        T = self._check_forward_mode_input_array(X)
        if tensor_rank == 0:
            #there is only 1 var name, so this will only run once
            for var_name in self.var_names:
                arg_vars[var_name] = X
        if tensor_rank == 1 and T == shape[0]:
            #there is only 1 var name, so this will only run once
            for var_name in self.var_names:
                arg_vars[var_name] = X.squeeze()
        elif tensor_rank == 1:
            for j, var_name in enumerate(sorted(self.var_names)):
                arg_vars[var_name] = X[j].squeeze()
        elif tensor_rank == 2:
            for j, var_name in enumerate(sorted(self.var_names)):
                arg_vars[var_name] = X[:,j].squeeze()
        return arg_vars

    def _check_forward_mode_input_dict(self, var_tbl: dict) -> int:
        """
        Check whether one forward mode input dict has elements of valid shape
        Returns inferred value of T
        """
        T: int = 1
        for var_name in var_tbl:
            # The bound value to this variable name
            val = var_tbl[var_name]
            # case 1: this is a scalar; T=1
            if isinstance(val, scalar_instance_types):
                t = 1
            # case 2: this is an array; calulate T
            elif isinstance(val, np.ndarray):
                t = self._calc_T_var(val)
            #case 3: throw an error
            else:
                raise ValueError(f'val={val} in var_tbl; {type(val)} not a recognized value type.')
            #update T
            if t > 1 and T == 1:
                T = t
            elif t not in (1,T):
                raise ValueError(f'Bound variable {var_name} has inconsistent shape')
        return T

    def _check_forward_mode_input_array(self, X: np.ndarray) -> int:
        """
        Check whether one forward mode input array is of valid shape
        Returns inferred value of T
        """
        # Find the length of each variable to infer T
        if not isinstance(X, np.ndarray):
            raise ValueError('X must be a numpy array, dict, or scalar')
        # Get the shape and tensor rank
        shape = X.shape
        tensor_rank = len(shape)
        T = 0
        # Only 1D and 2D arrays are supported
        if tensor_rank not in (0, 1, 2):
            raise ValueError(f'Shape of X = {X.shape}. Numpy array must be a 1D vector or 2D matrix')        
        if tensor_rank == 0:
            T = 1
        # If the input was a 1D vector, its length must EITHER (1) be T, or (2) m, with T == 1
        if tensor_rank == 1 and (shape[0] != self.m) and self.m != 1:
            raise ValueError(f'Error: X has shape {X.shape}, incompatible with m = {self.m} on fluxion.')
        # Return the value of T in this situation
        if tensor_rank == 1 and shape[0] == self.m:
            T = 1
        if tensor_rank == 1 and self.m == 1:
            T = shape[0]
        # If the input was a 2D vector, it must be of shape Txn
        if tensor_rank == 2 and (shape[1] != self.m):
            raise ValueError(f'Error: X has shape {X.shape}, incompatible with m = {self.m} on fluxion.')
        if tensor_rank == 2:
            T = shape[0]
        return T

    def _calc_T_var(self,X) -> int:
        """Calculate the number of samples, T, from the shape of X"""
        shape = X.shape
        tensor_rank: int = len(shape)
        if tensor_rank == 0:
            return 1
        if tensor_rank == 1:
            return shape[0]
        if tensor_rank == 2:
            if shape[1] > 1:
                raise ValueError('Initial value of a variable must have dimension T*1.')
            return shape[0]

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
class Const(Fluxion):
    """A function returning a constant; floats are implicitly promoted to instances of Const"""
    def __init__(self, a: scalar_type):
        # Initialize for a constant m=0 and n=1 (it depends on no variables)  
        self.m = 0
        self.n = 1
        self.var_names = {}
        # Constants are only floats (ints can be promoted)
        if not isinstance(a, scalar_instance_types):
            raise ValueError(f'Error: {a} is of type {type(a)}, not a scalar type (int or float).')
        # Promote an integer to a float if necessary
        self.a: float = float(a)

    def _forward_mode(self, *args):
        """Forward mode differentiation for a constant"""
        # Return the bound value and a derivative of zero regardless of the argumetns passed
        val: float = self.a
        diff: float = 0.0
        return (val, diff)

    def __repr__(self):
        return f'Const({self.a})'


# *************************************************************************************************
class Unop(Fluxion):
    """Abstract class embodying a unary operation"""
    def __init__(self, f: Fluxion):
        self.f = f
        self.m = f.m
        self.n = f.n
        self.var_names = f.var_names

class Power(Unop):
    """Raise a fluxion to the power p"""

    def __init__(self, f: Fluxion, p: float = 0.0):
        # Initialize the parent Unop class
        Unop.__init__(self,f)
        self.p = p

    def _forward_mode(self, *args):
        """Forward mode differentiation for a constant"""
        # Evaluate inner function self.f
        X: np.ndarray
        dX: np.ndarray
        X, dX = self.f._forward_mode(*args)
        # Alias the power to p for legibility
        p: float = self.p
        # The function value
        val = X ** p
        # The derivative
        diff = p * X ** (p-1) * dX
        return (val, diff)

    def __repr__(self):
        return f'Power({str(self.f)}, {self.p})'


# *************************************************************************************************
class Binop(Fluxion):
    """Abstract class embodying a binary operation"""

    def __init__(self, f: Fluxion, g: Fluxion):
        # Bind the input fluxions to members f and g of the binary operator
        self.f = f
        self.g = g       
        # Create the list of variable names
        var_names = f.var_names.copy()
        for v in g.var_names:
            if v in var_names:
                if (var_names[v] is None and g.var_names[v] is not None):
                    raise(KeyError("variable " + v + " has conflicting values"))
                elif (var_names[v] is not None and g.var_names[v] is None):
                    raise(KeyError("variable " + v + " has conflicting values"))
                elif (var_names[v] is not None and g.var_names[v] is not None):
                    if var_names[v].all() != g.var_names[v].all():
                        raise(KeyError("variable " + v + " has conflicting values"))
            else:
                var_names[v] = g.var_names[v]
        self.var_names = var_names
        self.m = len(self.var_names)
        # Check the shapes
        if f.n != g.n and min(f.n,g.n)>1:
            raise ValueError(f'In {self.__repr__()}, ms f.n={f.n} and g.n={g.n} must match for binary operation or one of them must be 1.')
        self.n = max(f.n,g.n)


class Addition(Binop):
    """Addition (sum) of two fluxions; h = f + g"""

    def __init__(self, f: Fluxion, g: Fluxion):
        # Initialize the parent Binop class
        Binop.__init__(self, f, g)

    def _forward_mode(self, *args):
        """Forward mode differentiation for a sum"""
        # (f+g)(x) = f(x) + g(x)
        f_val, f_diff = self.f._forward_mode(*args)
        g_val, g_diff = self.g._forward_mode(*args)
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

    def _forward_mode(self, *args):
        """Forward mode differentiation for a difference"""
        # (f-g)(x) = f(x) - g(x)
        f_val, f_diff = self.f._forward_mode(*args)
        g_val, g_diff = self.g._forward_mode(*args)
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

    def _forward_mode(self, *args):
        """Forward mode differentiation for a product"""
        # Product Rule of Calculus
        # https://en.wikipedia.org/wiki/Product_rule
        # (f*g)'(x) = f'(x) * g(x) + f(x) * g'(x)
        f_val, f_diff = self.f._forward_mode(*args)
        g_val, g_diff = self.g._forward_mode(*args)
        val = f_val * g_val
        # The function value is the product (elementwise)
        return (val, f_val * g_diff + f_diff * g_val)

    def __repr__(self):
        return f'Multiplication({str(self.f)}, {str(self.g)})'

class Division(Binop):
    """Division (quotient) of two fluxions; h = f * g"""

    def __init__(self, f: Fluxion, g: Fluxion):
        # Initialize the parent Binop class
        Binop.__init__(self, f, g)

    def _forward_mode(self, *args):
        # Quotient Rule of calculus
        # https://en.wikipedia.org/wiki/Quotient_rule
        # f(x) = g(x) / h(x),
        # f'(x) = (g('x)h(x) - g(x)h'(x)) / h(x)^2
        f_val, f_diff = self.f._forward_mode(*args)
        g_val, g_diff = self.g._forward_mode(*args)
        val = f_val / g_val
        return (val, (f_diff * g_val - f_val * g_diff) / (g_val * g_val))

    def __repr__(self):
        return f'Division({str(self.f)}, {str(self.g)})'

# *************************************************************************************************
class Var(Fluxion):
    """Class embodying the concept of a variable that is an input to a function"""
    def __init__(self, var_name: str, initial_value: Optional[np.ndarray]=None):
        """Variables must be instantiated with a name; binding an initial value is optional"""

        # Set the shape of the fluxion class
        self.m = 1
        self.n = 1

        if initial_value is not None:
            X = initial_value
            if not isinstance(X, value_instance_types):
                raise ValueError(f'Error: {X} of type {type(X)} is not a value type.  Must be int, float, or numpy array.')
            if isinstance(X, scalar_instance_types):
                # If X was a scalar, bind the value
                X = np.array(float(X))
                self.X = X
                self.T = 1
            else:
                # If X was an array, check the dimensions
                self.X = X
                self.T = self._calc_T_var(X)
        else:
            self.X = None
            self.T = 0

        #bind the variable
        self.var_name = var_name
        self.var_names = {var_name: self.X}

    def _forward_mode(self, *args):
        """Forward mode differentiation for variables"""
        # Parse arguments into two numpy arrays
        X: np.ndarray
        dX: np.ndarray
        X, dX = self._parse_dicts(*args)
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

    def _parse_dicts(self, *args) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse input arguments used in function evaluation.
        End result will be two arrays X of shape (m) or (T, m)
        Allowed input shapes are:
        (1) DICT:       dictionary mapping variable names to values
        (2) ARGS:       2 dictionaries mapping variable names to values
        """
        # initialize the X variable that will be returned
        X: np.ndarray
        dX: np.ndarray
        # get the number of arguments and inputs
        argc: int = len(args)

        if argc == 2:
            arg_vars = args[0]
            arg_seed = args[1]
            if arg_vars is None:
                return (None,None)
            X = self._parse_vars_tbl(arg_vars)
            dX = self._parse_seed_tbl(arg_seed)
            # if dX was the default dict, tile dX to match T
            if X.shape[0] > 1 and dX.shape[0] == 1:
                dX = np.tile(dX, (X.shape[0], 1))
            # Return the arrays of argument values
            return (X,dX)
        # If we reach here, the contract was broken
        msg = f'argc={argc}'
        for arg in args:
            msg += f'{arg}'
        raise ValueError(f'Contract broken. Unrecognized input type for Fluxion._parse_dicts.  Details: \n{msg}')

    def _parse_vars_tbl(self, var_tbl):
        """Parse a table of variable bindings (dictionary with key = variable name)"""

        # Find the length of each variable to infer T
        T = self._check_forward_mode_input_dict(var_tbl)
        # The shape of X based on T and m
        shape = (T, 1)

        # Initialize X to zeros in the correct shape        
        X = np.zeros(shape)
        X[:,0] = var_tbl[self.var_name]
        return X

    def _parse_seed_tbl(self, var_tbl):
        """Parse a table of variable bindings (dictionary with key = variable name)"""

        m: int = len(var_tbl)
        # Find the length of each variable to infer T
        T = self._check_forward_mode_input_dict(var_tbl)

        # The shape of X based on T and m
        shape = (T, m)

        # Initialize X to zeros in the correct shape        
        X = np.zeros(shape)
        # loop through the bound variables
        for j, var_name in enumerate(sorted(var_tbl)):
            if var_name in self.var_names:
                X[:,j] = var_tbl[var_name]
        return X

    def __repr__(self):
        return f'Var({self.var_name}, {self.X})'


# *************************************************************************************************
def Vars(*args):
    """Convenience method to return a tuple of variables"""
    return tuple(Var(x) for x in args)
