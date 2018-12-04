import numpy as np
import sys
import os
# Handle import of module fluxions differently if module
# module is being loaded as __main__ or a module in a package.
if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir('../..')
    import fluxions as fl
    os.chdir(cwd)
else:
    import fluxions as fl

# ***************************************************************************************
def report_success():
    """Report that a test was successful"""
    test_name = sys._getframe(1).f_code.co_name
    print(f'{test_name:25}: **** PASS ****')

# ***************************************************************************************
def test_elementary_functions():
    # Create a variable theta with angles from 0 to 360 degrees, with values in radians
    theta_val = np.expand_dims(np.linspace(0, 2*np.pi, 361), axis=1)
    theta = fl.Var('theta')

    # Scalar version
    f_theta = fl.sin(theta)
    assert f_theta.val({'theta':2}) == np.sin(2)
    assert f_theta.diff({'theta':2}) == np.cos(2)
    assert(str(f_theta)=="sin(Var(theta, None))")

    # Vector version
    f_theta = fl.sin(theta)
    assert np.all(f_theta.val({'theta':theta_val}) == np.sin(theta_val))
    assert np.all(f_theta.diff({'theta':theta_val}) == np.cos(theta_val))

    report_success()




# ***************************************************************************************
def test_basics_singlevar():
    theta_vec = np.expand_dims(np.linspace(-5,5,21) * np.pi, axis=1)

    #### TEST: passing in vector of values: "immediate" evaluation
    _cos, _dcos = fl.cos(theta_vec)()
    _sin, _dsin = fl.sin(theta_vec)()
    _tan, _dtan = fl.tan(theta_vec)()

    assert(all(_dcos == -_sin))
    assert(all(np.isclose(_tan, _sin/_cos)))


    #### TEST: passing a fluxion Var: "delayed" evaluation

    theta = fl.Var('theta')
    _cos = fl.cos(theta)
    _sin = fl.sin(theta)

    assert(np.all(np.isclose(_sin.diff({'theta':theta_vec}), np.cos(theta_vec))))
    assert(np.all(_cos.diff({'theta':theta_vec}) == -1*_sin.val({'theta':theta_vec})))


    #### TEST: basic functionality of other elementary functions
    # tan' = sec^2
    _dtan = (fl.tan(theta)).diff({'theta':theta_vec})
    _sec2 = ((1/fl.cos(theta))**2).val({'theta':theta_vec})
    assert(np.all(np.isclose(_dtan, _sec2)))

    # test Fluxions returns NaN as numpy does
    x = np.linspace(-5,5,21)
    _varx = fl.Var('x')
    _log = fl.log(_varx)
    # This test is tricky.  two subtleties:
    # (1) do everything under this with statement to catch runtime warnings about bad log inputs
    # (2) can't just compare numbers with ==; also need to compare whether they're both nans separately
    # this is because nan == nan returns FALSE in numpy!
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore')
        _log_fl = _log.val({'x':x})
        _log_np = np.expand_dims(np.log(x), axis=1)
        assert(np.all((_log_fl == _log_np) | (np.isnan(_log_fl) == np.isnan(_log_np))))
        _log_der_1 = _log.diff({'x':x})
        _log_der_2 = 1/_varx.val({'x':x})
        assert(np.all((_log_der_1 == _log_der_2)))

    _hypot, _hypot_der = fl.hypot(fl.sin(theta_vec).val(), fl.cos(theta_vec).val())()
    assert(np.all(_hypot == np.ones_like(theta_vec)))

    # test arccos vs. sec?

    report_success()


# ***************************************************************************************
def test_compositions():
    """ TEST: composition of elementary functions:
             (i) compositions of multiple elementary functions
            (ii) compositions of elementary functions & other ops (Fluxions)
    """
    theta_vec = np.expand_dims(np.linspace(-5,5,21) * np.pi, axis=1)

    # composition of 2 elementary functions:
    # (a) immediate evaluation
    val, diff = fl.log(fl.exp(theta_vec))()
    assert(np.all(val == theta_vec ))
    assert(np.all(np.isclose(diff, 1.0)))

    # (b) delayed evaluation
    logexp = fl.log(fl.exp(fl.Var('theta')))
    assert(np.all(logexp.val({'theta':theta_vec}) == theta_vec ))
    assert(np.all(np.isclose(logexp.diff({'theta':theta_vec}), np.ones_like(theta_vec))))

    # composition of elementary functions and basic ops (other Fluxions)
    # (a) immediate evaluation
    ans_1 = fl.cos(theta_vec).val()**2 - fl.sin(theta_vec).val()**2
    ans_2 = fl.cos(2*theta_vec).val()
    assert(np.all(np.isclose(ans_1, ans_2)))

    # (b) delayed evaluation
    theta = fl.Var('theta', theta_vec)
    f = fl.cos(theta)**2 - fl.sin(theta)**2
    assert(np.all(np.isclose(f.val({'theta':theta_vec}), fl.cos(2*theta_vec).val())))

    report_success()


# ***************************************************************************************
def test_basics_multivar():
    """ TEST: elementary functions of multiple variables """
    theta_vec = np.expand_dims(np.linspace(-5,5,21) * np.pi, axis=1)

    sin_z = fl.sin( fl.Var('x') * fl.Var('y') )
    assert(np.all(np.isclose(sin_z.val({'x':theta_vec, 'y':theta_vec}), np.sin(theta_vec**2))))

    #WHAT TO DO WHEN TOO FEW VARS PASSED? 
    # sin_z.val({'x':theta_vec})

    # HOW ARE PARTIALS EVALUATED?
    # sin_z.diff({'x': theta_vec})
    # sin_z.diff({'x': theta_vec, 'y': theta_vec*2})

    report_success()


# ***************************************************************************************
def test_trig():
    """Test the trigonometric functions"""
    theta = np.expand_dims(np.linspace(-5,5,21), axis=1)

    # Evaluation and derivative of sin(x), cos(x) and tan(x)
    assert(str(fl.sin)=="DifferentiableFunctionFactory(sin)")
    assert(str(fl.sin(0))=="FluxionResult(0.0,1.0)")
    _sin, _dsin = fl.sin(theta)()
    _cos, _dcos = fl.cos(theta)()
    _tan, _dtan = fl.tan(theta)()
    
    # Known answers
    # sin'(x) = cos(x)
    assert np.allclose(_dsin, _cos)
    
    # cos'(x) = -sin(x)
    assert np.allclose(_dcos, -1 * _sin)
    
    # tan'(x) = sec^2(x)
    _sec2 = 1.0 / np.cos(theta)**2
    assert np.allclose(_dtan, _sec2)
    
    report_success()


# ***************************************************************************************
def test_inverse_trig():
    """Test the inverse trigonometric functions"""
    theta = np.expand_dims(np.linspace(-0.9,0.9,21), axis=1)

    # Evaluate x, y and t at theta
    _y = np.sin(theta)
    _x = np.cos(theta)
    _t = np.tan(theta)

    # Evaluation and derivative of arcsin(x), arccos(x) and arctan(x)
    _arcsin, _darcsin = fl.arcsin(_y)()
    _arccos, _darccos = fl.arccos(theta)()
    _arctan, _darctan = fl.arctan(_t)()
    _arcsinh, _darcsinh = fl.arcsinh(theta)()
    _arccosh, _darccosh = fl.arccosh(theta+2)()
    _arctanh, _darctanh = fl.arctanh(theta)()

    # Ignore divide by zero errors (they are not real errors here)
    with np.errstate(divide='ignore'):
        # Known answers
        # arcsin'(y) = 1 / sqrt(1-y^2)
        assert np.allclose(_darcsin, 1.0 / np.sqrt(1.0 - _y**2))
        # arccos'(theta) = -1 / sqrt(1-theta^2)
        assert np.allclose(_darccos, -1.0 / np.sqrt(1.0 - theta**2))
        # arctan'(t) = 1 / (1+t^2)
        assert np.allclose(_darctan, 1.0 / (1.0 + _t**2))
        # arcsinh'(theta) = 1 / sqrt(1+theta^2)
        assert np.allclose(_darcsinh, 1.0 / np.sqrt(theta*theta + 1))
        # arccosh'(theta+2) = 1 / sqrt((theta+2)^2-1)
        assert np.allclose(_darccosh, 1.0 / np.sqrt((theta+2)**2 - 1))
        # arctanh'(theta) = 1 / (1-theta^2)
        assert np.allclose(_darctanh, 1.0 / (1.0 - theta*theta))

    report_success()


# ***************************************************************************************
def test_misc_trig():
    """Test the miscellaneous trigonometric functions"""
    theta = np.expand_dims(np.linspace(-0.9,0.9,21)*np.pi/2, axis=1)
    theta_rad = theta
    theta_deg = theta * (360.0 / (2 *np.pi))

    # Evaluate x, y and t at theta
    _y = np.sin(theta)
    _x = np.cos(theta)
    _t = np.tan(theta)

    # Evaluation and derivative of conversions between degrees and radians
    _degrees, _ddegrees = fl.degrees(theta_rad)()
    _radians, _dradians = fl.radians(theta_deg)()
    _rad2deg, _drad2deg = fl.rad2deg(theta_rad)()
    _deg2rad, _ddeg2rad = fl.deg2rad(theta_deg)()
    
    # Known answers for conversion between degrees and radians
    assert np.allclose(_degrees, theta_deg)
    assert np.allclose(_ddegrees, 360.0 / (2 *np.pi))
    assert np.allclose(_radians, theta_rad)
    assert np.allclose(_dradians, (2 *np.pi) / 360.0)

    assert np.allclose(_rad2deg, theta_deg)
    assert np.allclose(_drad2deg, 360.0 / (2 *np.pi))
    assert np.allclose(_deg2rad, theta_rad)
    assert np.allclose(_ddeg2rad, (2 *np.pi) / 360.0)

    # Evaluation and derivative of hypot(x, y), arctan2(y, x)
    _hypot, _dhypot = fl.hypot(_x, _y)()
    _arctan2, _darctan2 = fl.arctan2(_y, _x)()
    
    # Known answers for hypot function
    assert np.allclose(_hypot, 1.0)
    assert np.allclose(_dhypot, np.hstack([_x, _y]))
    
    # Known answers for arctan2 function
    assert np.allclose(_arctan2, theta)
    assert np.allclose(_darctan2, np.hstack([_x, -_y]))
    
    report_success()

# ***************************************************************************************
def test_hyperbolic():
    """Test hyperbolic functions"""
    x = np.linspace(-5, 5, 21)
    
    # Precompute e^x and e^(-x)
    exp_x = np.exp(x)
    exp_mx = np.exp(-x)
    
    # Evaluate sinh, cosh, tanh and their derivatives
    _sinh, _dsinh = fl.sinh(x)()
    _cosh, _dcosh = fl.cosh(x)()
    _tanh, _dtanh = fl.tanh(x)()

    # Known answers
    assert np.allclose(_sinh, (exp_x - exp_mx)/2.0)
    assert np.allclose(_dsinh, _cosh)
    assert np.allclose(_cosh, (exp_x + exp_mx)/2.0)
    assert np.allclose(_dcosh, _sinh)
    assert np.allclose(_tanh, (exp_x - exp_mx) / (exp_x + exp_mx))
    assert np.allclose(_dtanh, 1.0 / _cosh**2)    

    report_success()
    
# ***************************************************************************************
def test_exp_log():
    """Test exponents and logarithms"""
    x = np.linspace(-5, 5, 21)    
    exp_x = np.exp(x)
    y = exp_x

    # Evaluate exp(x), log(y)
    _exp, _dexp = fl.exp(x)()
    _log, _dlog = fl.log(y)()

    # Known answers
    assert np.allclose(_exp, exp_x)
    assert np.allclose(_dexp, exp_x)
    assert np.allclose(_log, x)
    assert np.allclose(_dlog, 1.0 / y)

    # Log base 2 and 10; exp base 2
    _log2, _dlog2 = fl.log2(y)()
    _log10, _dlog10 = fl.log10(y)()
    _exp2, _dexp2 = fl.exp2(x)()

    # Known answers
    assert np.allclose(_log2, x / np.log(2.0))
    assert np.allclose(_dlog2, 1.0 / y / np.log(2.0))
    assert np.allclose(_log10, x / np.log(10.0))
    assert np.allclose(_dlog10, 1.0 / y / np.log(10.0))    
    assert np.allclose(_exp2, 2.0**x)
    assert np.allclose(_dexp2, np.log(2.0) * (2.0**x) )

    # exponential minus 1, log plus 1
    _expm1, _dexpm1 = fl.expm1(x)()
    _log1p, _dlog1p = fl.log1p(y)()

    # Known answers
    assert np.allclose(_expm1, exp_x - 1.0)
    assert np.allclose(_dexpm1, exp_x)
    assert np.allclose(_log1p, np.log(1.0 + y))
    assert np.allclose(_dlog1p, 1.0 / (1.0 + y))

    # exponential minus 1, log plus 1
    _logaddexp, _dlogaddexp = fl.logaddexp(x,x)()
    _logaddexp2, _dlogaddexp2 = fl.logaddexp2(x,x)()
    assert(str(fl.logaddexp(fl.Var('x'),fl.Var('y')))=="logaddexp(Var(x, None),Var(y, None))")

    # Known answers
    assert np.allclose(_logaddexp, np.logaddexp(x,x))
    assert np.allclose(_dlogaddexp, np.vstack([0*y + 1/2, 0*y + 1/2]).T)
    assert np.allclose(_logaddexp2, np.logaddexp2(x,x))
    assert np.allclose(_dlogaddexp2, np.vstack([0*y + 1/2, 0*y + 1/2]).T)

    # forward mode
    f = fl.logaddexp(fl.Var('x'),fl.Var('y'))
    val, diff = f(0,0)
    assert(np.isclose(val,np.array([[0.69314718]])))
    assert(diff.all()== np.array([[0.5, 0.5]]).all())

    report_success()


# ***************************************************************************************
def test_misc():
    """Test miscellaneous functions"""
    x = np.linspace(-5, 5, 21)
    x2 = x * x
    x3 = x * x * x
    
    # Evaluate sqrt(x2), cbrt(x3)
    with np.errstate(divide='ignore'):
        _sqrt, _dsqrt = fl.sqrt(x2)()
        _cbrt, _dcbrt = fl.cbrt(x3)()
        _square, _dsquare = fl.square(x)()
    
    # Known answers
    with np.errstate(divide='ignore'):
        assert np.allclose(_sqrt, np.abs(x))
        assert np.allclose(_dsqrt, 0.5 / _sqrt)
        assert np.allclose(_cbrt, x)
        assert np.allclose(_dcbrt, (1.0/3.0) / (x*x))
        assert np.allclose(_square, x2)
        assert np.allclose(_dsquare, 2*x)

    report_success()

# *************************************************************************************************
test_elementary_functions()
test_basics_singlevar()
test_compositions()
test_basics_multivar()
test_trig()
test_inverse_trig()
test_misc_trig()
test_hyperbolic()
test_exp_log()
test_misc()
