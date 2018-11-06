import numpy as np
import os
# module is being loaded as __main__ or a module in a package.
if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir('../')
    import fluxions as fl
    os.chdir(cwd)
else:
    import fluxions as fl


# Create a variable, x
x = fl.Var('x', 1.0)

#f0 = x - 1
f0 = x - 1
# assert(f0.val({'x':1}) == 0)
# assert(f0.diff({'x':1}) == 1)
var_tbl = {'x':1}
seed_tbl = {'x':1}
val, diff = f0.forward_mode(var_tbl, seed_tbl)
assert val == 0
assert diff == 1
assert repr(f0) == "Subtraction(Var(x, 1.0), Const(1.0))"

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
assert(f1({'x':2}) == (10.0, np.array([5.])))
assert repr(f1) == "Multiplication(Var(x, 1.0), Const(5.0))"

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
assert repr(f3) == "Division(Addition(Var(x, 1.0), Const(1.0)), Multiplication(Var(x, 1.0), Var(x, 1.0)))"

# f4(x) = (1 + 5x)/(x * x)
f4 = (1 + 5 * x) / (x * x)
f4.set_var_names('x')
assert(f4.val({'x':2}) == 2.75)
assert(f4.diff({'x':2}) == -1.5)

#    # Take a power
#    f5 = fl.Power(x, 2)
#    f5.set_var_names('x')
#    assert(f5.val(8) == 64)
#    assert(f5.diff(8) == 16)
#    assert(f5.val(8, 3) == 512)
#    assert(f5.diff(8, 3) == 3*64)
#    assert(f5() == (1.0, 2.0))
#    assert(f5(1) == (1.0, 2.0))
#    assert(f5({'x':1}) == (1.0, 2.0))
#    assert repr(f5) == "Power(x, 2)"

#check assignment
a = fl.Fluxion()
assert a.is_node()
b = fl.Unop(a)
b.set_var_name('x')
assert b.is_node()
c = fl.Var('x')
assert(c.diff(0) == 1)
assert(c.diff({'x':1}) == 1)
assert(c.diff({'x':1},{'x':2}) == 2)
assert(c.diff({'x':1,'y':1},{'x':2,'y':1}) == np.array([[2, 0]])).all()
c.set_val(0)
assert(c(1)==(1, np.array([1])))

#check division
f6 = 1/x
f6 = 1/x
assert(f6.val({'x':1,'y':1}) == 1)
assert(f6.diff({'x':1,'y':1}) == np.array([[-1.,  0.]])).all()

#check subtraction and division
f7 = (1 - x + 1 - 1) / ((x * x)/1)
f7.set_var_names('x')
assert(f7.val({'x':2}) == -0.25)
assert(f7.diff({'x':2}) == 0)

