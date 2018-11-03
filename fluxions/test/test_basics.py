import pytest

# Use the __init__ file in this directory to set the path so this runs from the command line
import __init__ as init
init.set_path()
# Now ready to import fluxions - it will be on the search path
import fluxions as fl


# *************************************************************************************************
def test_basic_usage():
    # f(x) = 5x
    f_x = 5 * fl.Var('x')
    assert(f_x.eval({'x':2}) == 10)
    assert(f_x.diff({'x':2}) == 5)

    # f(x) = 1 + (x * x)
    f_x = 1 + fl.Var('x') * fl.Var('x')
    assert(f_x.eval({'x':2}) == 5)
    assert(f_x.diff({'x':3}) == 6)  

    # f(x) = (1 + x)/(x * x) 
    f_x = (1 + fl.Var('x')) / (fl.Var('x') * fl.Var('x'))
    assert(f_x.eval({'x':2}) == 0.75)
    assert(f_x.diff({'x':2}) == -0.5)

    # f(x) = (1 + 5x)/(x * x) 
    f_x = (1 + 5 * fl.Var('x')) / (fl.Var('x') * fl.Var('x'))
    assert(f_x.eval({'x':2}) == 2.75)
    assert(f_x.diff({'x':2}) == -1.5)

# Run the test
test_basic_usage()
# Report results
print(f'test_basic_usage: **** PASS ****')
