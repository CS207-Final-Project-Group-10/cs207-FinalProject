import numpy as np
import sys
import os
# Handle import of module fluxions differently if module
# is being loaded as __main__ or a module in a package.
if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir('../..')
    import fluxions as fl
    os.chdir(cwd)
else:
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

    #f0 = x - 1
    f0 = x - 1
    assert(f0.val({'x':1}) == 0)
    assert(f0.diff({'x':1}) == 1)
    var_tbl = {'x':1}
    seed_tbl = {'x':1}
    val, diff = f0(var_tbl, seed_tbl)
    assert val == 0
    assert diff == 1
    assert repr(f0) == "Subtraction(Var(x, 1.0), Const(1.0))"

    # f1(x) = 5x
    f1 = 5 * x
    # Evaluate f1(x) at the bound value of x
    assert(f1() == (5.0, 5.0))
    assert(f1(None)==(5.0, 5.0))
    assert(f1(1,1) == (5.0, 5.0))
    assert(f1(np.array(1),np.array(1)) == (5.0, 5.0))
    # Evaluate f1(x) using function calling syntax
    assert(f1(2) == (10.0, 5.0))
    # Evaluate f1(x) using dictionary binding syntax
    assert(f1.val({'x':2}) == 10)
    assert(f1.diff({'x':2}) == 5)
    assert(f1({'x':2}) == (10.0, np.array([5.])))
    assert repr(f1) == "Multiplication(Var(x, 1.0), Const(5.0))"

    # f2(x) = 1 + (x * x)
    f2 = 1 + x * x
    assert(f2(4.0) == (17.0, 8.0))
    assert(f2.val({'x':2}) == 5)
    assert(f2.diff({'x':3}) == 6)

    # f3(x) = (1 + x)/(x * x)
    f3 = (1 + x) / (x * x)
    assert(f3.val({'x':2}) == 0.75)
    assert(f3.diff({'x':2}) == -0.5)
    assert repr(f3) == "Division(Addition(Var(x, 1.0), Const(1.0)), Multiplication(Var(x, 1.0), Var(x, 1.0)))"

    # f4(x) = (1 + 5x)/(x * x)
    f4 = (1 + 5 * x) / (x * x)
    assert(f4.val({'x':2}) == 2.75)
    assert(f4.diff({'x':2}) == -1.5)

    # Take a power
    f5 = fl.Power(x, 2)
    assert(f5.val(8) == 64)
    assert(f5.diff(8) == 16)
    assert(f5() == (1.0, 2.0))
    assert(f5(1) == (1.0, 2.0))
    assert(f5({'x':1}) == (1.0, 2.0))
    assert repr(f5) == "Power(Var(x, 1.0), 2)"

    #check assignment
    a = fl.Fluxion()
    b = fl.Unop(a)
    c = fl.Var('x')
    assert(c.diff(0) == 1)
    assert(c.diff({'x':1}) == 1)
    assert(c.diff({'x':1},{'x':2}) == 2)
    assert(np.array_equal(c.diff({'x':1,'y':1},{'x':2,'y':1}), np.array([[2., 0.]])))
    assert(c(1)==(1, np.array([1])))

    #check division
    f6 = 1/x
    assert(f6.val({'x':1,'y':1}) == 1)
    assert(np.array_equal(f6.diff({'x':1,'y':1}),np.array([[-1.,  0.]])))

    #check subtraction and division
    f7 = (1 - x + 1 - 1) / ((x * x)/1)
    assert(f7.val({'x':2}) == -0.25)
    assert(f7.diff({'x':2}) == 0)

    # check negation
    f8 = -x
    assert(f8.val({'x':2}) == -2)
    assert(f8.diff({'x':2}) == -1)

    y = fl.Var('y')
    f9 = -(x * y)
    assert(f9.val({'x':-2, 'y':3}) == 6)
    val, diff = f9(1,1,1,1)
    assert(val == np.array([[-1.]]))
    assert(val == np.array([[-1., -1.]])).all()

# Report results
report_success()


def test_basics_vectors():
    """Test using Fluxions objects with vector inputs"""
    # Create some vectors
    n = 10
    xs = np.expand_dims(np.linspace(0,1,num=n), axis=1)
    ys = np.linspace(1,2,num=n)
    ys_ex = np.expand_dims(np.linspace(1,2,num=n), axis=1)

    # Create variables x and y bound to vector values
    x = fl.Var('x', xs)
    y = fl.Var('y', ys)

    # f1(x) = 5x
    f1 = 5 * x
    assert(f1.val(xs) == 5*xs).all()
    assert(f1.diff({'x':xs}) == 5.0*np.ones(np.shape(xs))).all()

    # f2(x) = 1 + (x * x)
    f2 = 1 + x * x
    assert(f2.val({'x':xs}) == 1 + np.power(xs,2)).all()
    assert(f2.diff({'x':xs}) == 2.0*xs).all()

    # f3(y) = (1 + y)/(y * y)
    f3 = (1 + y) / (y * y)
    assert(f3.val({'y':ys}) == np.divide(1+ys_ex,np.power(ys_ex,2))).all()
    assert np.isclose(f3.diff({'y':ys_ex}), np.divide(-2-ys_ex,np.multiply(np.power(ys_ex,2),ys_ex))).all()

    # f(x) = (1 + 5x)/(x * x)
    f4 = (1 + 5*x) / (x * x)
    assert(f4.val({'x':ys}) == np.divide(1+5*ys_ex,np.power(ys_ex,2))).all()
    assert np.isclose(f4.diff({'x':ys}),np.divide(-2-5*ys_ex,np.multiply(np.power(ys_ex,2),ys_ex))).all()

    # f5(x,y) = 5x+y
    f5 = 5 * x + y
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
    f7 = 3 * x + 2 * y + z
    var_tbl_scalar = {'x':1,'y':1,'z':1}
    assert(f7.val(var_tbl_scalar) == 6)
    assert(f7.diff(var_tbl_scalar) == np.array([3, 2, 1])).all()
    var_tbl_vector = {'x':xs,'y':xs,'z':xs}
    assert(f7.val(var_tbl_vector) == 3*xs + 2*xs + xs).all()
    assert(f7.diff(var_tbl_vector) == np.asarray([np.array([3, 2, 1])]*10)).all()
    var_tbl_vector = {'z':xs}
    f7.val(var_tbl_vector)
    assert(f7.val(var_tbl_vector) == 3*xs + 2*xs + xs+2).all()

    # f(x,y,z) = (3x+2y+z)/xyz
    f8 = (x * 3 + 2 * y + z)/(x * y * z)
    assert(f8.val(var_tbl_scalar) == 6)
    assert(f8.diff(var_tbl_scalar) == np.array([-3., -4., -5.])).all()
    # Rebind 'x', 'y', ans 'z' to the values in ys (slightly tricky!)
    var_tbl_vector = {'x':ys,'y':ys,'z':ys}
    assert(f8.val(var_tbl_vector) == (3*ys_ex + 2*ys_ex + ys_ex)/(ys_ex*ys_ex*ys_ex)).all()
    assert np.isclose(f8.diff(var_tbl_vector),
                     np.transpose([-3*ys/np.power(ys,4), -4*ys/np.power(ys,4), -5*ys/np.power(ys,4)])).all()

    #f(x,y) = xy
    f9 = y*x
    assert(f9.val({'x':0,'y':0,'z':1})==0).all()
    assert(f9.diff({'x':0,'y':0,'z':1})==np.asarray([np.array([0, 0, 0])])).all()

# Report results
report_success()

# Run the test
test_basic_usage()
test_basics_vectors()