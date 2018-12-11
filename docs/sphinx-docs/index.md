## Fluxions Documentation

## Introduction

* [Click here to see the full API Reference](source/fluxions.html)

Differential calculus was invented as a formal branch of mathematics in the middle of the 17th century independently by Isaac Newton and Gottfried Leibniz.  Newton's orginal term for a derivative with respect to time was a "fluxion," which gave its name to his last book, *Method of Fluxions*, published posthumously. Newton used differential calculus to solve the problem of the motion of the planets around the sun, and it has proven to be an essential tool for the sciences ever since.  In the modern era, essentially all scientific calculations of interest are performed on computers.  There are many scenarios where we know how to compute a function of interest and would like to efficiently evaluate its derivative(s).  The canonical example is root finding.  If we know a function's derivative, we can iteratively improve from a starting guess until we find a root using a simple procedure, Newton's Method.  Many phenomena of interest in physics and other sciences can be described as differential equations (either ODEs or PDEs).  The ability to efficiently evaluate derivatives is crucial in numerically solving these systems.

In recent years, Machine Learning has been a hot research area.  Solving ML problems often hinges on the ability to train a neural network using some form of a gradient descent algorithm.  As the name suggests, this algorithm requires the ability to evaluate not just the function (here the output of the neural net) but also its first derivative, which in practice is typically the change in the error metric with respect to the parameter values.  These neural networks are massively complex functions that can have millions of parameters.  A procedure of numerically differentiating the network by shifting the value of each parameter has cost that scales linearly with the number of parameters.  Some form of automatic differentiation is vital to the practical use of neural nets.  One of the most successful machine learning libraries is TensorFlow by Google, which at its core is an enterprise class automatic differentiation library.

## How to use `fluxions`

### Installation Instructions

#### For end users:
The `fluxions` package is available on PyPI. Before installing ensure you have a Python3 environment with numpy installed.

If you choose to install the package into a virtual environment, do so by first setting up the environment in your desired directory as follows:

```console
pip3 install virtualenv
virtualenv -p python3 venv
source venv/bin/activate

pip3 install numpy
```

Once you have an appropriate environment set up, you can install the `fluxions` package with the following command:

```console
pip3 install fluxions
```

#### For developers:

Clone the [git repository](https://github.com/CS207-Final-Project-Group-10/cs207-FinalProject) to a location of your choice.

Ensure you have a Python3 environment available. If you want to use a virtual environment, execute the following code in the cloned directory:

```console
pip3 install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
```

Finally, install the requirements

```console
pip3 install -r requirements.txt
```

We use pytest for testing. In order to run the tests, execute the following from the root of the cloned directory:

```console
pytest
```


### Brief How-To and Demo

Users can build up complex functions using a combination of numerical constants, `fluxions` `Var` objects, and `fluxions` elementary function objects like `sin` and `exp`, as well as the standard set of basic Python operators including +, -, /, \*, etc. (The complete list of functions available can be found under the *Elementary Functions* section, below.) 

The function and its derivative(s) can then be evaluated by calling function's `val` or `diff` methods. When calling `val`, a dictionary mapping keys (i.e. variable names) to numeric values must be passed if the `Var` instance(s) in the function have not already been bound to a numeric value. When passing values to `val`, the variable names must correspond to names given to the `Var` object(s) used to create the function. The same holds when differentiating a function by calling its `diff` method. But when differentiating a function of multiple variables, in addition to passing the dictionary mapping variable names to numeric values, the user must also specify the seed values from which to calculate the derivative. 

All values will be returned from both the `val` and `diff` functions as `numpy` arrays that match the dimensions of the input values.

For example, to find the value and derivative of the logistic function evaluated at x=0:

```python
>>> import fluxions as fl
>>> 
>>> # Treating x as a symbolic variable:
>>> x = fl.Var('x')
>>> f = 1 / (1 + fl.exp(-x))
>>> f.val({'x':0})

array([0.5])

>>> f.diff({'x':0})

array([0.25])

>>> # Alternatively, treating x as a numeric constant:
>>> x = fl.Var('x', 0)
>>> f = 1 / (1 + fl.exp(-x))
>>> f.val(), f.diff()

(0.5, 0.25)
```

Or, to evaluate the logistic function at a vector of input values, simply pass a `numpy` array instead of a scalar:

```python
>>> # continuing previous example...
>>> import numpy as np
>>> 
>>> f.val({'x':np.array([-1,0,1])})

array([0.26894142, 0.5       , 0.73105858])

>>> f.diff({'x':np.array([-1,0,1])})
 
array([[0.19661193, 0.25      , 0.19661193]])

```

Considering instead a function of multiple variables:

```python
>>> x = fl.Var('x')
>>> y = fl.Var('y')
>>> f = fl.log(x * y)
>>> 
>>> # evaluating f(x,y) at (x,y)=(2,0.5)
>>> f.val({'x':2, 'y':0.5})

array([0.])

>>> # partial with respect to x, evaluated at (x,y)=(2,0.5)
>>> f.diff({'x':2, 'y':0.5}, {'x':1, 'y':0})

array([0.5])

>>> # partial with respect to y, evaluated at (x,y)=(2,0.5)
>>> f.diff({'x':2, 'y':0.5}, {'x':0, 'y':1})

array([2.])

```

## Background
Calculus gives us a few simple rules for computing the derivative of a function composed of two more elementary subfunctions.  Most central to Automatic Differentiation is the Chain Rule of calculus, which states that the derivative of a composition is the product of two derivatives:
.. math::
  f'(u(x)) = f'(u(x) \cdot u'(x)
The chain rule works in multiple dimensions.  If :math:`f` is a function from $\mathbb{R}^n$ to $\mathbb{R}^m$, its derivative is an $m$ x $n$ matrix called the Jacobian.  The chain rule in the multidimensional case tells us to take the matrix product of an $m$ x $r$ matrix and an $r$ x $n$ matrix to compute the derivative.

The essential idea of Automatic Differentiation is that computing the  derivative of any function can be reduced to evaluating and differentiating a sequence of simple subfunctions. These may be strung together in a complex graph with multiple steps, but each step will be a basic operation.  For the evaluation of a mathematical function, each step will consist of either a "primitive" or built-in function (e.g. +, -, x, /, sqrt, log, exp, pow, etc.) or a composition of these primitives.  As long as we know how to evaluate both the function f(x) and its derivative f'(x) at each step, we can trace the calculation through the computation graph and also keep track of the first derivative.

The forward mode of Automatic Differentiation takes 3 arguments: a function, $f(x)$, a vector $x \in \mathbb{R}^n$ at which to evaluate the function $f(x)$, and a vector of 'seed' values for the derivatives, $dx\in \mathbb{R}^
n$. Following the standard order of operations, the innermost elementary function calls of $f$ are evaluated at x and dx. As above, as long as we know how to evaluate both the elementary function f(x) and its derivative f'(x) at each step, pass the results of each to enclosing functions and apply the chain rule to keep track of the first derivative. Once all function calls are complete, the algorithm returns the values $f(x)$ and $f'(x)$.

## Software organization

### Directory structure

The `fluxions` package has the following directory structure:

  ```
     fluxions/
           fluxions/
                 __init__.py
                 elementary_functions.py
                 fluxion_node.py             
                 test/
                      __init__.py
                      test_basics.py
                      test_elementary_functions.py
           .gitignore
           .travis.yml
           LICENSE.txt
           README.md
           setup.cfg
           setup.py
           requirements.txt
           example_newtons_method.py
  ```

`fluxion_node.py` contains the definitions for the core `Fluxion` class, the unary (`Unop`) and binary (`Binop`) operator abstract classes, and all classes that implement the operators, e.g. `Addition`, `Multiplication`. (These classes are discussed further under the *Implementation Details* section, below). The abstract `DifferentiableFunction` and `DifferentiableInnerFunction` classes, as well as static objects that implement specific elementary functions, are found in `elementary_functions.py`.

#### Package tests
All tests are located in the directory *fluxions/test*. There are two primary test suites. To run all of these tests using pytest, simply navigate to the root package directory and execute:

```console
pytest fluxions/
```
   
Or to run just one of the test suites,

```console
pytest fluxions/test/test_basics.py
```

#### Example driver
The short Python program `newtons_method.py` found in the package root directory is a trivial example of a "driver" that uses the `fluxions` package to solve a problem. The program illustrates how the package can be used to implement a basic Newton's Method solver that works on functions from $\mathbb{R}$ to $\mathbb{R}$. It takes a differentiable function (Fluxion instance) as its first input, and uses forward-mode differentiation to return both the value and exact first derivative.

## Implementation details

### Core Data Structures

To optimize computational efficiency, `fluxions` is built around the `NumPy` package; all numerical data including inputs, outputs, and intermediate calculations are stored in `numpy` arrays.  The only `fluxions` class that persists numeric data (stored as a numpy.ndarray) is the `Var` class; when it persists data (either at initialization or by calling the `set_val` method), that is stored as an array. 

The package makes light use of some built-in Python data structures and their methods for parts of the code that are not critical to performance. In particular, Python dictionaries are used to specify the values and seeds with which to evaluate and differentiate functions. The keys in these dictionaries are the variable names, while the values are the numeric values (whether scalar or numpy arrays) bound to that variable.

### Core Classes

#### *Fluxion*

All of the classes in this package are ultimately derived from the `Fluxion` base class, consistent with the fundamental idea in Automatic Differentiation that constants, variables, and elementary functions can all be treated as differentiable functions upon which complex functions are built. This class defines a common interface for the input and output behavior of all inheriting function classes.

Together with the basic operator classes described below, the Fluxion implements operator overloading for the Python symbolic operators. When one of these operators is performed on a `Fluxion` object, the object dispatches suitable calls to instances of the class `Binop` or `Unop`.

The Fluxion class also defines the interface through which users can interact with functions -- namely, by calling the function's `val`, `diff`, or 'jacobian' methods. The `val` method corresponds to function evaluation; no derivatives are calculated.  The `diff` method corresponds to performing a single pass of forward mode evaluation with one seed value.  Both of these methods take as inputs `*args`, allowing them to be called with alternative arguments.  The `val` method accepts a single numpy array `X` or a dictionary `var_tbl` in which each variable name has been mapped to either a scalar value (`float` or `int`) or a numpy array.  When an array is passed, it can have shape `(n)` or `(T, n)`.  The inputs for `diff` need to be sufficient to resolve both the input array $X$ as well as the seed value.  The analogous footprints include a pair of arrays, which we would label $X$ and $dX$; or a pair of dictionaries, which we could label `var_tbl` and `seed_tbl`.

Finally, the `parse_args_val` and `parse_args_forward_mode` Fluxion methods implement the machinery for handling various input types the user may supply.  Localizing this code in the Fluxion class at the very top of the inheritance hierarchy allows us to avoid duplicating it elsewhere. 

When a Fluxion instance is created, it sets the tensor size attributes $m$, $n$ and $T$, and sets the `name` and `var_names` attributes (see below for more details).

#### *Unop, Const and Var*

The `Unop` class embodies the concept of a unary operator.  Descendants of Unop have non-trivial implementations of `val` and `diff`.  Two key examples of Unops include `Const` and `Var`. `Const` is a constant, depending on no variables and having a derivative of zero.  `Var` is a logical entity for a variable.  The statement `x = fl.Var('x')` creates an instance of a Fluxion Var class.  The local name `x` in our Python program has `x.name = 'x'` and `x.variable_names= ['x']`.  This permits us to then create another Fluxion e.g. `f = x * x + 5` which behaves as we expect it to.  Other examples of classes that inherit from Unop include the `Power` and `ElementaryFunction` classes.

#### *Binop, Addition, Subtraction, Multiplication and Division*

The `Binop` class embodies the concept of a binary operator.  All of the work of performing the calculations of a binary operation are performed in the children of `Binop`.  The `Binop` itself is quite minimal; it binds the two input fluxions to attributes called `f` and `g`.  It also sets a preliminary list of variable names by merging the names of variables for `f` and `g`, suppressing duplicates. 

We can understand all four basic arithmetic operations through one example, `Muliplication`.  The primary methods of the Multiplication class are `val` and `forward_mode`.  The evaluation method `val` dispatches calls to `f` and `g` and returns their product.  The forward_mode method builds up `f_val, f_diff` by calling `f.forward_mode`, and similarly for `g`.  Then it applies the product rule of calculus.  It is worth reiterating here that the intermediate results returned in `f_val, f_diff, g_val, g_diff` are all numpy arrays, so the computation `f_val * g_diff + f_diff * g_val` is performed with maximum efficiency.

#### *Elementary Functions*

Please see the major section below for a discussion of the implementation for elementary functions.

### Important Attributes

Every `Fluxion` represents a function from $\mathbb{R^n}$ to $\mathbb{R^n}$.  These should be invariant after the Fluxion is instantiated, though this is not currently enforced.  The sizes are integer attributes named `m` and `n` on the Fluxion class.  A Fluxion also has a notion of the number of sample points `T`. This is analogous to basic vectorization in `numpy`.  For example, if you call `np.sin` on a numpy array of shape (100,), you get back another array of shape (100,).  You only pay the overhead of calling the function and doing Python "unboxing" once.

Every Fluxion has a name; these are not expected to be unique.  Variables also have names.  This makes it possible for users to pass inputs with dictionaries, and makes the output of the `__repr__` function comprehensible.  Every Fluxion also an attribute called `var_names`, which is a list of the variables that it depends on (i.e. its inputs).  This attribute controls the interpretation of input presented to a Fluxion for evaluation.  For example, the Fluxions implementation extending the `numpy` built-in function `atan2` has variable names `['y', 'x']`.  This also controls the order of columns in the Jacobian.

### External Dependencies

The minimial set of external dependencies for `fluxions` are listed in `requirements.txt`. These dependencies include `numpy`, of which the `fluxions` package is essentially an extension, as well as `pytest` and `pytest-cov` for testing.

### Elementary Functions

The elementary function objects are all specific instances of the generic `DifferentiableFunction` class, which itself inherits from the Unop (and thus the Fluxions) class. These functions represent basic mathematical functions that are analytically differentiable.  They are all defined around corresponding `numpy` implementations, which perform the actual numerical calculations.

The elementary functions supported are as follows:

​ Trigonometric Functions: `sin, cos, tan`

​ Inverse Trigonometric Functions: `arcsin, arccos, arctan`

​ Miscellaneous Trigonometric Functions: `hypot, arctan2, degrees, radians, deg2rad, rad2deg`

​ Hyperbolic Functions: `sinh, cosh, tanh, arcsinh, arccosh, arctanh`

​ Exponents & Logarithms: `exp, expm1, log, log10, log2, log1p, logaddexp, logaddexp2`

While the elementary function objects are instances of the `DifferentiableFunction` class, they contain an additional `DifferentiableInnerFunction` object. This inner object is also a `Unop` that represents a single node on the calculation graph. This inner object is necessitated by the Chain Rule of calculus: for a composition of functions $f(g(x))$, the derivative of the outer function with respect to $x$ is $df/dg*dg/dx$. It is therefore necessary for the outer function (f) to store an instance of the inner function (g), so that when a user requests the derivative of f is requested, differentiation can first be recursively propagated to the inner function before computing the derivative of the outermost function.

The `DifferentiableInnerFunction` class has one input, a Fluxion `f` that can be either a value type or another Fluxion.  Its `__init__` method binds this other Fluxion, plus two callables `func` and `deriv`.  As an example, when we construct a node `elem_func_node = fl.sin(fl.Var('x'))`for $f(x) = sin(x)$, the input `elem_func_node.f`is an instance of `Var` with name 'x', and the two callables are `node.func = np.sin` and `node.deriv = np.cos`.

The `DifferentiableFunction` class is what is exposed to users. It is essentially a factory for producing instances of `DifferentiableInnerFunction`.  The `__init__` method associates one of these factories with the two callables for the function and its derivatives, plus the tensor sizes and names. That the elementary functions are all created from the DifferentiableFunction class allows users to create additional differentiable functions as desired.


## Future Plans
