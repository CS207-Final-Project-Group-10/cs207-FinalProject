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
    theta_val = np.linspace(0, 2*np.pi, 361)
    theta = fl.Var('theta')
    
    # Scalar version
    f_theta = fl.sin(theta)
    assert f_theta.val({'theta':2}) == np.sin(2)
    assert f_theta.diff({'theta':2}) == np.cos(2)

    # Vector version
    f_theta = fl.sin(theta)
    assert np.all(f_theta.val({'theta':theta_val}) == np.sin(theta_val))
    assert np.all(f_theta.diff({'theta':theta_val}) == np.cos(theta_val))

    report_success()




# ***************************************************************************************
def __test_basics_singlevar():
    theta_vec = np.linspace(-5,5,21) * np.pi

    #### TEST: passing in vector of values: "immediate" evaluation
    _cos, _dcos = fl.cos(theta_vec)
    _sin, _dsin = fl.sin(theta_vec)
    _tan, _dtan = fl.tan(theta_vec)

    assert(all(_dcos == -_sin))
    assert(all(_tan == _sin/_cos))


    #### TEST: passing a fluxion Var: "delayed" evaluation

    theta = fl.Var('theta')
    _cos = fl.cos(theta)
    _sin = fl.sin(theta)

    assert(all( _cos.diff({'theta':theta_vec}) == np.cos(theta_vec)))
    assert(all( _cos.diff({'theta':theta_vec}) == -1*_sin.val({'theta':theta_vec}) ))


    #### TEST: basic functionality of other elementary functions
    # tan' = sec^2
    _dtan = (fl.tan(theta)).diff({'theta':theta_vec})
    _sec2 = ((1/fl.cos(theta))**2).diff({'theta':theta_vec})
    assert(all(_dtan == _sec2))

    # test Fluxions returns NaN as numpy does
    x = np.linspace(-5,5,21)
    _varx = fl.Var('x')
    _log = fl.log(_varx)
    assert(all( _log.val({'x':x}) == np.log(x) ))
    assert(all( _log.val({'x':x}) == 1/_varx({'x':x})))

    assert(all( fl.hypot(fl.sin(theta_vec), fl.cos(theta_vec)) == np.ones_like(theta_vec) ))
    
    # test arccos vs. sec?

    report_success()


# ***************************************************************************************
def __test_compositions():
    """ TEST: composition of elementary functions:
             (i) compositions of multiple elementary functions
            (ii) compositions of elementary functions & other ops (Fluxions)
    """
    theta_vec = np.linspace(-5,5,21) * np.pi

    # composition of 2 elementary functions: 
    # (a) immediate evaluation
    val_diff_result = fl.log(fl.exp(theta_vec))
    assert(all( val_diff_result[0] == np.ones_like(theta_vec) ))
    assert(all( val_diff_result[1] == theta_vec )) 

    # (b) delayed evaluation
    logexp = fl.log(fl.exp(fl.Var('theta')))
    assert(all( logexp.val({'theta':theta_vec}) == np.ones_like(theta_vec) ))
    assert(all( logexp.diff({'theta':theta_vec}) == theta_vec ))

    # composition of elementary functions and basic ops (other Fluxions)
    # (a) immediate evaluation
    assert(all( fl.cos(theta_vec)**2 - fl.sin(theta_vec)**2 == fl.cos(2*theta_vec) ))

    # (b) delayed evaluation
    f = fl.cos(theta)**2 - fl.sin(theta)**2
    assert(all( f.val({'theta':theta_vec}) == fl.cos(2*theta_vec) ))


    report_success()


# ***************************************************************************************
def __test_basics_multivar():
    """ TEST: elementary functions of multiple variables """ 
    theta_vec = np.linspace(-5,5,21) * np.pi

    sin_z = fl.sin( fl.Var('x') * fl.Var('y') )
    assert(all( sin_z.val({'x':theta_vec, 'y':theta_vec}) == np.sin(theta_vec)**2 ))

    #WHAT TO DO WHEN TOO FEW VARS PASSED? 
    # sin_z.val({'x':theta_vec})

    # HOW ARE PARTIALS EVALUATED?
    # sin_z.diff({'x': theta_vec})
    # sin_z.diff({'x': theta_vec, 'y': theta_vec*2})

    report_success()




# *************************************************************************************************
test_elementary_functions()

#test_basics_singlevar()
#test_compositions()
#test_basics_multivar()

