# Example "driver" script demonstrating use of the fluxions package
# Use case is a Newton's Method solver



# import fluxions as fl
# from fluxions.fluxions import Fluxion
# from fluxions.elementary_functions import exp


def newtons_method_scalar(f, x: float, tol: float =1e-8) -> float:
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
        x += dx
    # Return x and the number of iterations required
    return x, i


# Use this newton's method solver to find the root to equation
# e^x = 10x
