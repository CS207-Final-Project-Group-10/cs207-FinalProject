import numpy as np
from numpy import pi
from fluxions import Unop, Var

class sin(Unop):
    """
    Fluxions implementation of sin(x)
    The derivative of sin(x) is cos(x)
    """

    def val(self, args=None):
        return np.sin(self.val.args)
    
    def diff(self, args=None):
        return np.sin(self.val.args)
    
# Create a variable theta with angles from 0 to 360 degrees, with values in radians
theta = Var('theta', np.linspace(0, 2*pi, 361)

y = sin.val(theta)