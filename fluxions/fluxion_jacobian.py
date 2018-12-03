import numpy as np
from importlib import util
# Handle import of classes in fluxion_node differently based on import as module or run from test
if util.find_spec("fluxions") is not None:
    from fluxions import Fluxion
else:
    from fluxion_node import Fluxion


def _check_input_vals(v_mapping):
    # wraps values in numpy arrays, as required for evaluation by fluxions,
    # if values in v_mapping are lists or scalars
    #
    # TODO: 
    #    (1) PERHAPS THIS SHOULD BE DONE DOWNSTREAM (WITHIN FLUXION OBJECT)
    #    (2) Check type of each value in dict to ensure homogeneity?
    #    (3) Any more efficient way of doing this conversion for all dict values?
    keys = list(v_mapping.keys())
    vals = list(v_mapping.values())
    if type(vals[0]) == np.ndarray:
        return v_mapping
    elif type(vals[0]) == list:
        return {key:np.array(val) for key, val in zip(keys, vals)}
    else:
        # values are scalars
        return {key:np.array([val]) for key, val in zip(keys, vals)}


"""
    >>> x = fl.Var('x')
    >>> y = fl.Var('y')
    >>> fl.jacobian([x*y, x**2, x+y], ['x','y'], {'x':2, 'y':3})
    array([[3., 2.],
           [4., 0.],
           [1., 1.]])

    # Jacobian of a scalar function (= transpose of its gradient)
    >>> z = fl.Var('z')    
    >>> jacobian([2*x + x*y**3 + fl.log(z)], ['z','y','x'], {'x':2, 'y':3, 'z':4})
    array([0.25, 54, 29]

    # Jacobian of a function from R^2 -> R^3, T=1
    >>> J = jacobian([x*y, x**2, x+y], ['x','y'], {'x':1, 'y':2})
    >>> J.shape
    (3,2)
    
    # Jacobian of a function from R^2 -> R^3, T=4
    >>> v_mapping = {'x':list(np.linspace(1,4,4)), 'y':list(2*np.ones(4))}
    >>> J = jacobian([x*y, x**2, x+y], ['x','y'], v_mapping)
    >>> J.shape
    (3,2,4)

"""
def jacobian(f, v, v_mapping):
    """
        f: single fluxion object or an array or list of fluxions, representing a scalar or vector function
        v: vector of variables in f with respect to which the Jacobian will be calculated
        v_mapping: dict mapping variables in f to scalar or vector of values

    """    
    # make sure f is in the form np.array([fl1, ...])
    if isinstance(f, Fluxion):
        f = [f]
    f = np.asarray(f)
    v = np.asarray(v)
    v_mapping = _check_input_vals(v_mapping)

    m = len(v)
    n = len(f)
    T = len(list(v_mapping.values())[0]) # number of values per variable
    
    J = np.zeros((m,n,T))
    for i, f_i in enumerate(f):
        seed = dict.fromkeys(v, 1)
            
        dfi_dvj = f_i.diff(v_mapping, seed)
        J[:,i,:] = dfi_dvj.T

    return J.squeeze()