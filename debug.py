# from fluxion_node import Fluxion, Unop
import numpy as np
import fluxions as fl
from fluxions import Fluxion, Var, DifferentiableFunctionNode, sin

x = fl.Var('x', 2)
y = fl.Var('y', 3)
f1 = x * y
f1.set_var_names(['x', 'y'])

ans_pos = f1(2, 3)
# print(f'f1(2,3) = {ans_pos}, expected 6.')
assert ans_pos[0] == 6

ans_dict = f1({'x': 2, 'y': 3})
# print(f'f1(var_tbl) = {ans_dict}, expected 6.')
assert ans_dict[0] == 6



# Create a variable theta with angles from 0 to 360 degrees, with values in radians
theta_val = np.linspace(0, 2*np.pi, 361)
theta = fl.Var('theta')

# Scalar version
f_theta = fl.sin(theta)
var_tbl = {'theta':2}
assert f_theta.val(var_tbl) == np.sin(2)
assert f_theta.diff(var_tbl) == np.cos(2)

# Vector version
f_theta = fl.sin(theta)
assert np.all(f_theta.val({'theta':theta_val}) == np.sin(theta_val))
assert np.all(f_theta.diff({'theta':theta_val}) == np.cos(theta_val))


#    # f = sin(x * y)
#    f = sin(f1)
#    
#    
sin_x = sin(x)
#sin_x_eval = sin_x()
sin_x.val()
