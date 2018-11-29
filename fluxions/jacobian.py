
import numpy as np

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


    >>> r = fl.Var('r')
    >>> theta = fl.Var('theta')
    >>> F = [r*cos(theta), r*sin(theta)] 
    # J(F) = [[cos(theta), -r*sin(theta)],[sin(theta), r*cos(theta)]]

"""
def jacobian(f, vars, vars_mapping):
    """
        f: single fluxion object or an array or list of fluxions, representing a scalar or vector function
        vars: vector of variables in f with respect to which the Jacobian will be calculated
        vars_mapping: dict mapping variables in f to scalar or vector of values

    """
    
    f = np.asarray(f)
    vars = np.asarray(vars)
    
    J = np.zeros((len(f.ravel()),len(vars)))
    for i, f_i in enumerate(f):
        for j, v_j in enumerate(vars):
            # make seed dict
            seed = dict.fromkeys(vars, 0)
            seed[v_j] = 1

            J[i,j] = f_i.diff(vars_mapping, seed)

    return J
