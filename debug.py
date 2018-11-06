# from fluxion_node import Fluxion, Unop
import numpy as np
import fluxions as fl
from fluxions import Fluxion, Var, differentiable_function, sin

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


# f = sin(x * y)
f = sin(f1)

theta_val = np.linspace(0, 2*np.pi, 361)
theta = fl.Var('theta', theta_val)
cos_theta = fl.cos(theta)

sin_x = sin(x)
# sin_x_eval = sin_x()
sin_x.val()
