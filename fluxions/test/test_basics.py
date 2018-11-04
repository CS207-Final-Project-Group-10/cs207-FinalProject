import pytest

# Use the __init__ file in this directory to set the path so this runs from the command line
#import __init__ as init
#init.set_path()
# Now ready to import fluxions - it will be on the search path
from fluxions import fluxions as fl


# *************************************************************************************************
def test_basic_usage():
    # f(x) = 5x
    f_x = 5 * fl.Var('x')
    assert(f_x.val({'x':2}) == 10)
    assert(f_x.diff({'x':2}) == 5)
    assert(repr(f_x)=="Multiplication(Var(x, None), Const(5))")

    # f(x) = 1 + (x * x)
    f_x = 1 + fl.Var('x') * fl.Var('x')
    assert(f_x.val({'x':2}) == 5)
    assert(f_x.diff({'x':3}) == 6)

    # f(x) = (1 + x)/(x * x)
    f_x = (1 + fl.Var('x')) / (fl.Var('x') * fl.Var('x'))
    assert(f_x.val({'x':2}) == 0.75)
    assert(f_x.diff({'x':2}) == -0.5)

    # f(x) = (1 + 5x)/(x * x)
    f_x = (1 + 5 * fl.Var('x')) / (fl.Var('x') * fl.Var('x'))
    assert(f_x.val({'x':2}) == 2.75)
    assert(f_x.diff({'x':2}) == -1.5)

    # Create a variable, x
    x = fl.Var('x', 1.0)

    # f(x) = 5x
    f1 = 5 * x
    f1.set_var_names('x')
    # Evaluate f1(x) at the bound value of x
    # Evaluate f1(x) using function calling syntax
    assert(f1(2) == (10.0, 5.0))
    # Evaluate f1(x) using dictionary binding syntax
    assert(f1.val({'x':2}) == 10)
    assert(f1.diff({'x':2}) == 5)

    # f(x) = 1 + (x * x)
    f2 = 1 + x * x
    f2.set_var_names('x')
    assert(f2(4.0) == (17.0, 8.0))
    assert(f2.val({'x':2}) == 5)
    assert(f2.diff({'x':3}) == 6)

    # f(x) = (1 + x)/(x * x)
    f3 = (1 + x) / (x * x)
    f3.set_var_names('x')
    assert(f3.val({'x':2}) == 0.75)
    assert(f3.diff({'x':2}) == -0.5)

    # f(x) = (1 + 5x)/(x * x)
    f4 = (1 + 5 * x) / (x * x)
    f4.set_var_names('x')
    assert(f4.val({'x':2}) == 2.75)
    assert(f4.diff({'x':2}) == -1.5)

    # Vector Examples
    n = 10
    xs = np.linspace(0,1,num=n)
    ys = np.linspace(1,2,num=n)

    # f(x) = 5x
    f_x = 5 * fl.Var('x')
    assert(f_x.val({'x':xs}) == 5*xs).all()
    assert(f_x.diff({'x':xs}) == 5.0*np.ones(np.shape(xs))).all()

    # f(x) = 1 + (x * x)
    f_x = 1 + fl.Var('x') * fl.Var('x')
    assert(f_x.val({'x':xs}) == 1 + np.power(xs,2)).all()
    assert(f_x.diff({'x':xs}) == 2.0*xs).all()

    # f(x) = (1 + x)/(x * x)
    f_x = (1 + fl.Var('x')) / (fl.Var('x') * fl.Var('x'))
    assert(f_x.val({'x':ys}) == np.divide(1+ys,np.power(ys,2))).all()
    assert (np.linalg.norm(f_x.diff({'x':ys}) - np.divide(-2-ys,np.multiply(np.power(ys,2),ys))) < 10**(-15))

    # f(x) = (1 + 5x)/(x * x)
    f_x = (1 + 5*fl.Var('x')) / (fl.Var('x') * fl.Var('x'))
    assert(f_x.val({'x':ys}) == np.divide(1+5*ys,np.power(ys,2))).all()
    assert (np.linalg.norm(f_x.diff({'x':ys}) - np.divide(-2-5*ys,np.multiply(np.power(ys,2),ys))) < 10**(-14))

    # f(x,y) = 5x+y
    f_xy = 5 * fl.Var('x') + fl.Var('y')
    assert(f_xy.val({'x':2,'y':3}) == 13)
    assert(f_xy.diff({'x':2,'y':3}) == np.array([5, 1])).all()
    assert(f_xy.val({'x':xs,'y':xs}) == 5*xs + xs).all()
    assert(f_xy.diff({'x':xs,'y':xs}) == np.asarray([np.array([5, 1])]*n)).all()

    # f(x,y) = 5xy
    f_xy = 5 * fl.Var('x') * fl.Var('y')
    assert(f_xy.val({'x':2,'y':3}) == 30)
    assert(f_xy.diff({'x':2,'y':3}) == np.array([15, 10])).all()
    assert(f_xy.val({'x':xs,'y':xs}) == np.multiply(5*xs,xs)).all()
    assert(f_xy.diff({'x':xs,'y':xs}) == np.transpose([5*xs,5*xs])).all()

    # f(x,y,z) = 3x+2y+z
    xs = np.linspace(0,1)
    f_xyz = 3 * fl.Var('x') + 2 * fl.Var('y') + fl.Var('z')
    assert(f_xyz.val({'x':1,'y':1,'z':1}) == 6)
    assert(f_xyz.diff({'x':1,'y':1,'z':1}) == np.array([3, 2, 1])).all()
    assert(f_xyz.val({'x':xs,'y':xs,'z':xs}) == 3*xs + 2*xs + xs).all()
    assert(f_xyz.diff({'x':xs,'y':xs,'z':xs}) == np.asarray([np.array([3, 2, 1])]*50)).all()

    # f(x,y,z) = (3x+2y+z)/xyz
    xs = np.linspace(0,1)
    f_xyz = (3 * fl.Var('x') + 2 * fl.Var('y') + fl.Var('z'))/(fl.Var('x') * fl.Var('y') * fl.Var('z'))
    assert(f_xyz.val({'x':1,'y':1,'z':1}) == 6)
    assert(f_xyz.diff({'x':1,'y':1,'z':1}) == np.array([-3., -4., -5.])).all()
    assert(f_xyz.val({'x':ys,'y':ys,'z':ys}) == (3*ys + 2*ys + ys)/(ys*ys*ys)).all()
    assert (np.linalg.norm(f_xyz.diff({'x':ys,'y':ys,'z':ys}) \
                           -np.transpose([-3*ys/np.power(ys,4), -4*ys/np.power(ys,4), -5*ys/np.power(ys,4)])) < 10**(-14))


# Run the test
test_basic_usage()
# Report results
print(f'test_basic_usage: **** PASS ****')
