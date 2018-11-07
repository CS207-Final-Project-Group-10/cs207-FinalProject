import numpy as np
import os
# module is being loaded as __main__ or a module in a package.
if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir('../')
    import fluxions as fl
    os.chdir(cwd)
    from fluxions import DifferentiableFunction, DifferentiableInnerFunction, sin
else:
    import fluxions as fl
    from fluxions import DifferentiableFunction, DifferentiableInnerFunction, sin




theta = np.linspace(-5,5,21) * np.pi
# _sin, _dsin = sin(theta)

f = theta
func = np.sin
deriv = np.cos
func_name = 'sin'
var_names = ['x']
m = 1
n= 1

