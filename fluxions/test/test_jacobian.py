import numpy as np
import os
# Handle import of module fluxions differently if module
# module is being loaded as __main__ or a module in a package.
if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir('../..')
    import fluxions as fl
    from fluxions import jacobian
    os.chdir(cwd)
else:
    import fluxions as fl
    from fluxions import jacobian

# *************************************************************************************************

x = fl.Var('x')
y = fl.Var('y')
z = fl.Var('z') 


def test_vectorization():
    global x, y, z

    # Jacobian of a function from R^2 -> R^3, T=1 is squeezed to remove T dimension
    J = jacobian([x*y, x**2, x+y], ['x','y'], {'x':1, 'y':2})
    assert(J.shape == (3,2))
    assert(J == np.array([[2, 1],
                          [2, 0],
                          [1, 1]])).all()
    
    # Jacobian of a function from R^2 -> R^3, T=4
    v_mapping = {'x':list(np.linspace(1,4,4)), 'y':list(2*np.ones(4))}
    J = jacobian([x*y, x**2, x+y], ['x','y'], v_mapping)
    assert(J.shape == (3,2,4))
    assert(np.all(J[:,:,0] == np.array([[2, 1],
                                        [2, 0],
                                        [1, 1]])))
    assert(np.all(J[:,:,3] == np.array([[2, 4],
                                        [8, 0],
                                        [1, 1]])))



def test_jacobian_dims():
    # test different possible combinations of function input and output dimensions
    global x, y, z

    J = jacobian([x*y, x**2, x+y], ['x','y'], {'x':2, 'y':3})
    assert(np.all(J == np.array([[3, 2],
                                 [4, 0],
                                 [1, 1]])))

    # Jacobian of a (1x1) function is the same as its derivative
    #F = fl.sin(fl.log(x**x))
    F = fl.sin(fl.log(x))
    J = jacobian(F, ['x'], {'x':2})
    assert(np.all(J == F.diff({'x':2})))


    # Jacobian of a scalar (3x1) function (= transpose of its gradient)
    J = jacobian([2*x + x*y**3 + fl.log(z)], ['z','y','x'], {'x':2, 'y':3, 'z':4})
    assert(np.all(J == np.array([0.25, 54, 29])))

    # partials with respect to only one variable -> Jacobian is still mxn
    J = jacobian([2*x + z*y**3 + fl.log(z)], ['z'], {'x':2, 'y':3, 'z':4})
    assert(np.all(J == np.array(27.25)))
    # NOTE: SHOULD THIS BE np.array([27.25])?


    J = jacobian([x+y, y], ['x','y','z'], {'x':2, 'y':3, 'z':1})
    assert(np.all(J == np.array([[1, 1, 0],
                                 [0, 1, 0]])))



    r = fl.Var('r')
    theta = fl.Var('theta')
    F = [r*fl.cos(theta), r*fl.sin(theta)] 
    # J(F) = [[cos(theta), -r*sin(theta)],[sin(theta), r*cos(theta)]]
    #J4 = jacobian([x = r*fl.sin(theta)*fl.cos(phi), r*fl.sin(theta)*fl.sin(phi), ])



def test_others():
    # test different input v_mapping types (list, scalar)
    pass


# run tests
test_vectorization()
test_jacobian_dims()
