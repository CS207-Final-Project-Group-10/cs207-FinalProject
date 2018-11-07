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



n=11
xs = np.linspace(0,1,num=n)
ys = np.linspace(1,2,num=n)

x = fl.Var('x', xs)
y = fl.Var('y', xs)

var_tbl_scalar = {'x':2, 'y':3}
var_tbl_vector = {'x':xs, 'y':ys}


# f(x,y) = 5xy
f6 = 5 * x * y
assert(f6.val(var_tbl_scalar) == 30)
# assert(f6.diff(var_tbl_scalar, {'x':1}) == np.array([15, 10])).all()
args = (var_tbl_scalar, {'y':1})
assert(f6.diff(var_tbl_scalar, {'x':1}) == np.array([15])).all()
assert(f6.diff(var_tbl_scalar, {'y':1}) == np.array([10])).all()
assert(f6.val(var_tbl_vector) == np.multiply(5*xs,ys)).all()
assert(f6.diff(var_tbl_vector, {'x':1}) == 5*ys).all()
assert(f6.diff(var_tbl_vector, {'y':1}) == 5*xs).all()
