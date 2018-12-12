import fluxions as fl
import numpy as np
import scipy.optimize as op

#wrapper for putting fluxions into a format compatible with scipy
def FluxionWrapper(F: fl.Fluxion):
    class wrapper():
        def val(x):
            return F.val(x)[0]
        def diff(x):
            return F.diff(x)[0]
    return wrapper

#steepest descent
def steepest_descent(x, f: fl.Fluxion, tol: float =1e-8):
    w = FluxionWrapper(f)
    xs = np.zeros([2001,2])
    xs[0]=x
    for i in range(0,2000):
        x = xs[i]
        grad = -w.diff(x)
        alpha = op.line_search(w.val,w.diff,x,grad)
        xs[i+1]= x + alpha[0]*grad
        stepsize = np.linalg.norm(xs[i+1]-xs[i])
        if stepsize < tol:
            break
    return (xs[0:i+1,:], i+1, stepsize, xs[i])

#newton's method
def newtons_method(x, F, dF, tol: float =1e-8):
    var_dict = {}
    var_list = []
    for f in dF:
        for v in sorted(F.var_names):
            var_dict[v] = F.var_names[v]
            if v not in var_list:
                var_list.append(v)
    xs = np.zeros([2001,2])
    xs[0]=x
    var_dict[var_list[0]]=x[0]
    var_dict[var_list[1]]=x[1]
    for i in range(0,2000):
        x = xs[i]
        var_dict[var_list[0]]=x[0]
        var_dict[var_list[1]]=x[1]
        grad = -F.diff(var_dict).squeeze()
        xs[i+1]= x + np.linalg.solve(fl.jacobian(dF, var_list, var_dict),grad).squeeze()
        stepsize = np.linalg.norm(xs[i+1]-xs[i])
        if stepsize < tol:
            break
    return (xs[0:i+1,:],i, stepsize, xs[i])

#BFGS
def bfgs(x,f: fl.Fluxion, tol: float =1e-8):
    w = FluxionWrapper(f)
    B = np.eye(2)
    xs = np.zeros([2001,2])
    xs[0]=x
    for i in range(0,2000):
        x = xs[i]
        grad = -w.diff(x)
        s = np.linalg.solve(B,grad)
        xs[i+1]= x + s
        y = w.diff(xs[i+1])+grad
        B = B + np.outer(y,y.T)/np.dot(y,s)-np.dot(np.dot(B,np.outer(s,s.T)),B)/np.dot(s.T,np.dot(B,s))
        stepsize = np.linalg.norm(s)
        if stepsize < tol:
            break
    return (xs[0:i+1,:],i + 1, stepsize, xs[i])