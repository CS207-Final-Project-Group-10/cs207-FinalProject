import fluxions as fl
import pytest
import numpy as np
import sys

def report_success():
    """Report that a test was successful"""
    test_name = sys._getframe(1).f_code.co_name
    print(f'{test_name:25}: **** PASS ****')

def test_basic_elementary_functions():     
    # Create a variable theta with angles from 0 to 360 degrees, with values in radians
    theta_val = np.linspace(0, 2*np.pi, 361)
    theta = fl.Var('theta', theta_val)

    # Scalar version
    y, dy_dx = fl.sin(2)
    assert y == np.sin(2)
    assert dy_dx == np.cos(2)

    # Vector version
    y, dy_dx = fl.sin(theta)
    assert np.all(y == np.sin(theta_val))
    assert np.all(dy_dx == np.cos(theta_val))

    report_success()