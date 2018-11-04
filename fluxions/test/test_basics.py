import pytest
import numpy as np
import sys

import fluxions as fl

# *************************************************************************************************
def report_success():
    """Report that a test was successful"""
    test_name = sys._getframe(1).f_code.co_name
    print(f'{test_name:25}: **** PASS ****')
    

def test_basic_usage():
    """Test basic usage of Fluxions objects"""
    # Create a variable, x
    x = fl.Var('x', 1.0)

    # f1(x) = 5x
    f1 = 5 * x
    f1.set_var_names('x')
    # Evaluate f1(x) at the bound value of x
    assert(f1() == (5.0, 5.0))
    # Evaluate f1(x) using function calling syntax
    assert(f1(2) == (10.0, 5.0))
    # Evaluate f1(x) using dictionary binding syntax
    assert(f1.val({'x':2}) == 10)
    assert(f1.diff({'x':2}) == 5)

    # f2(x) = 1 + (x * x)
    f2 = 1 + x * x
    f2.set_var_names('x')
    assert(f2(4.0) == (17.0, 8.0))
    assert(f2.val({'x':2}) == 5)
    assert(f2.diff({'x':3}) == 6)

    # f3(x) = (1 + x)/(x * x)
    f3 = (1 + x) / (x * x)
    f3.set_var_names('x')
    assert(f3.val({'x':2}) == 0.75)
    assert(f3.diff({'x':2}) == -0.5)

    # f4(x) = (1 + 5x)/(x * x)
    f4 = (1 + 5 * x) / (x * x)
    f4.set_var_names('x')
    assert(f4.val({'x':2}) == 2.75)
    assert(f4.diff({'x':2}) == -1.5)

    # Take a power
    f5 = fl.Power(x, 2)
    f5.set_var_names('x')
    assert(f5.val(8) == 64)
    assert(f5.diff(8) == 16)
    assert(f5.val(8, 3) == 512)
    assert(f5.diff(8, 3) == 3*64)    

    # Report results
    report_success()


def test_basics_vectors():
    """Test using Fluxions objects with vector inputs"""
    # Create some vectors
    n = 10
    xs = np.linspace(0,1,num=n)
    ys = np.linspace(1,2,num=n)

    # Create a variable, x bound to vector value
    x = fl.Var('x', xs)

    # f1(x) = 5x
    f1 = 5 * x
    f1.set_var_names('x')
    assert(f1.val(xs) == 5*xs).all()
    assert(f1.diff({'x':xs}) == 5.0*np.ones(np.shape(xs))).all()

    # f2(x) = 1 + (x * x)
    f2 = 1 + x * x
    f2.set_var_names('x')
    assert(f2.val({'x':xs}) == 1 + np.power(xs,2)).all()
    assert(f2.diff({'x':xs}) == 2.0*xs).all()

    # f3(y) = (1 + y)/(y * y)
    y = fl.Var('y', ys)
    f3 = (1 + y) / (y * y)
    f3.set_var_names('y')
    assert(f3.val({'y':ys}) == np.divide(1+ys,np.power(ys,2))).all()
    assert np.isclose(f3.diff({'y':ys}), np.divide(-2-ys,np.multiply(np.power(ys,2),ys))).all()
    

    # f(x) = (1 + 5x)/(x * x)
    f4 = (1 + 5*x) / (x * x)
    f4.set_var_names('x')
    assert(f4.val({'x':ys}) == np.divide(1+5*ys,np.power(ys,2))).all()
    assert np.isclose(f4.diff({'x':ys}),np.divide(-2-5*ys,np.multiply(np.power(ys,2),ys))).all()

    # f5(x,y) = 5x+y
    f5 = 5 * x + y
    f5.set_var_names(['x', 'y'])
    var_tbl_scalar = {'x':2, 'y':3}
    var_tbl_vector = {'x':xs, 'y':xs}
    assert(f5.val(var_tbl_scalar) == 13)
    assert(f5.diff(var_tbl_scalar) == np.array([5, 1])).all()
    assert(f5.val(var_tbl_vector) == 5*xs + xs).all()
    assert(f5.diff(var_tbl_vector) == np.asarray([np.array([5, 1])]*n)).all()

    # f(x,y) = 5xy
    f6 = 5 * x * y
    assert(f6.val(var_tbl_scalar) == 30)
    assert(f6.diff(var_tbl_scalar) == np.array([15, 10])).all()
    assert(f6.val(var_tbl_vector) == np.multiply(5*xs,xs)).all()
    assert(f6.diff(var_tbl_vector) == np.transpose([5*xs,5*xs])).all()

    # f(x,y,z) = 3x+2y+z
    z = fl.Var('z')
    xs = np.linspace(0,1)
    f7 = 3 * x + 2 * y + z
    f7.set_var_names(['x', 'y', 'z'])
    var_tbl_scalar = {'x':1,'y':1,'z':1}
    assert(f7.val(var_tbl_scalar) == 6)
    assert(f7.diff(var_tbl_scalar) == np.array([3, 2, 1])).all()
    var_tbl_vector = {'x':xs,'y':xs,'z':xs}
    assert(f7.val(var_tbl_vector) == 3*xs + 2*xs + xs).all()
    assert(f7.diff(var_tbl_vector) == np.asarray([np.array([3, 2, 1])]*50)).all()

    # f(x,y,z) = (3x+2y+z)/xyz
    f8 = (3 * x + 2 * y + z)/(x * y * z)
    f8.set_var_names(['x', 'y', 'z'])
    assert(f8.val(var_tbl_scalar) == 6)
    assert(f8.diff(var_tbl_scalar) == np.array([-3., -4., -5.])).all()
    # Rebind 'x', 'y', ans 'z' to the values in ys (slightly tricky!)
    var_tbl_vector = {'x':ys,'y':ys,'z':ys}
    assert(f8.val(var_tbl_vector) == (3*ys + 2*ys + ys)/(ys*ys*ys)).all()
    assert np.isclose(f8.diff(var_tbl_vector), 
                      np.transpose([-3*ys/np.power(ys,4), -4*ys/np.power(ys,4), -5*ys/np.power(ys,4)])).all()

    # Report results
    report_success()

# Run the test
test_basic_usage()
test_basics_vectors()