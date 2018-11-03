import pytest
import fluxions

def test_basic_usage():
    # f(x) = 5x
    f_x = 5 * Var('x')
    assert(f_x.eval({'x':2}) == 10)
    assert(f_x.diff({'x':2}) == 5)

    # f(x) = 1 + (x * x)
    f_x = 1 + Var('x') * Var('x')
    assert(f_x.eval({'x':2}) == 5)
    assert(f_x.diff({'x':3}) == 6)  

    # f(x) = (1 + x)/(x * x) 
    f_x = (1 + Var('x')) / (Var('x') * Var('x'))
    assert(f_x.eval({'x':2}) == 0.75)
    assert(f_x.diff({'x':2}) == -0.5)

    # f(x) = (1 + 5x)/(x * x) 
    f_x = (1 + 5 * Var('x')) / (Var('x') * Var('x'))
    assert(f_x.eval({'x':2}) == 2.75)
    assert(f_x.diff({'x':2}) == -1.5)

