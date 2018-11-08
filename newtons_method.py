# Example "driver" script demonstrating use of the fluxions package
# Use case is a Newton's Method solver

import fluxions as fl


def newtons_method_scalar(f: fl.Fluxion, x: float, tol: float =1e-8) -> float:
    """Solve the equation f(x) = 0 for a function from R->R using Newton's method"""
    max_iters: int = 100
    for i in range(max_iters):
        # Evaluate f(x) and f'(x)
        y, dy_dx = f(x)
        # Is y within the tolerance?
        if abs(y) < tol:
            break
        # Compute the newton step
        dx = -y / dy_dx
        # update x
        x += float(dx)
    # Return x and the number of iterations required
    return x, i


# Use this newton's method solver to find the root to equation
# e^x = 10x
x = fl.Var('x')
f = fl.exp(x) - 10*x

root, iters = newtons_method_scalar(f, 0.0)
f_root = float(f.val(root))
print(f'Solution of exp(x) == 10x by Newton''s Method:')
print(f'Solution converged after {iters} iterations.')
print(f'x={root:0.8f}, f(x) = {f_root:0.8f}')
