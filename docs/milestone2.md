# Milestone 2

## Introduction
Differential calculus was invented as a formal branch of mathematics in the middle of the 17th century independently by Isaac Newton and Gottfried Leibniz.  Newton's orginal term for a derivative with respect to time was a "fluxion," which gave its name to his last book, *Method of Fluxions*, published posthumously. Newton used differential calculus to solve the problem of the motion of the planets around the sun, and it has proven to be an essential tool for the sciences ever since.  In the modern era, essentially all scientific calculations of interest are performed on computers.  There are many scenarios where we know how to compute a function of interest and would like to efficiently evaluate its derivative(s).  The canonical example is root finding.  If we know a function's derivative, we can iteratively improve from a starting guess until we find a root using a simple procedure, Newton's Method.  Many phenomena of interest in physics and other sciences can be described as differential equations (either ODEs or PDEs).  The ability to efficiently evaluate derivatives is crucial in numerically solving these systems.
In recent years, Machine Learning has been a hot research area.  Solving ML problems often hinges on the ability to train a neural network using some form of a gradient descent algorithm.  As the name suggests, this algorithm requires the ability to evaluate not just the function (here the output of the neural net) but also its first derivative, which in practice is typically the change in the error metric with respect to the parameter values.  These neural networks are massively complex functions that can have millions of parameters.  A procedure of numerically differentiating the network by shifting the value of each parameter has cost that scales linearly with the number of parameters.  Some form of automatic differentiation is vital to the practical use of neural nets.  One of the most successful machine learning libraries is TensorFlow by Google, which at its core is an enterprise class automatic differentiation library.

## Background
Calculus gives us a few simple rules we can apply to compute the derivative of a function that is the result of a combination of two more simple functions.  Calculus rules include the sum, product, and quotient of two functions.  The most important calculus rule in Automatic Differentiation is the Chain Rule of calculus.  This states that the derivative of a composition is the product of two derivatives.
$$f'(u(x)) = f'(u(x) \cdot u'(x)$$
The chain rule works in multiple dimensions.  If $f$ is a function from $\mathbb{R}^n$ to $\mathbb{R}^m$, its derivative an $m$ x $n$ matric called the Jacobian.  The chain rule in the multidimensional case tells us to take the matrix product of an $m$ x $r$ matrix and an $r$ x $n$ matrix to compute the derivative.

The essential idea of Automatic Differentiation is that any computation performed by a computer will consist of a set of basic steps, e.g. function evaluations.  These may be strung together in a complex graph with multiple steps, but each step will be a basic operation.  For the evaluation of a mathematical function, each step will consist of either a "primitive" or built-in function (e.g. +, -, x, /, sqrt, log, exp, pow, etc.) or a composition of these primitives.  As long as we know how to evaluate both the function f(x) and its derivative f'(x) at each step, we can trace the calculation through the computation graph and also keep track of the first derivative.  This is the idea at the heart of Automatic Differentiation.

The forward mode of Automatic Differentiation takes 3 arguments, a function, $f(x)$, a vector at which to evaluate the function,  $x \in \mathbb{R}^n$, and a vector of 'seed' values for the derivatives, $dx\in \mathbb{R}^
n$. Following the standard order of operations, the innermost elementary function calls of $f$ are evaluated at x and dx. As above, as long as we know how to evaluate both the elementary function f(x) and its derivative f'(x) at each step, pass the results of each to enclosing functions and apply the chain rule to keep track of the first derivative. Once all function calls are complete, the algorithm returns the values $f(x)$ and $f'(x)$.


## How to use `fluxions`

### Installation Instructions

#### For end users:
The `fluxions` package is available on Test PyPI. Before installing ensure you have a Python3 environment with numpy installed.

If you choose to install the package into a virtual environment, do so by first setting up the environment in your desired directory as follows:

```console
pip3 install virtualenv
virtualenv -p python3 venv
source venv/bin/activate

pip3 install numpy
```

Once you have an appropriate environment set up, you can install the `fluxions` package with the following command:

```console
pip3 install --index-url https://test.pypi.org/simple/ fluxions
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
pytest fluxions/
```


### Basic demo

The intended way of using the fluxions package is by only accessing values that are exposed at the package level.

Consider importing the fluxions package as follows:

```python
import fluxions as fl
```




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
  ```

`fluxion_node.py` contains the definitions for the core `Fluxion` class, the unary (`Unop`) and binary (`Binop`) operator abstract classes, and all classes that implement the operators, e.g. `Addition`, `Multiplication`. (These classes are discussed further under the *Implementation Details* section, below). The abstract `DifferentiableFunction` and `DifferentiableInnerFunction` classes, as well as static objects that implement specific elementary functions, are found in `elementary_functions.py`.

Tests for each of these two modules 

    * Basic modules and what they do
    * Where do tests live?  How are they run?  How are they integrated?


## Implementation details
### Core Data Structures

All numerical data...

### Core Classes

xxx

### Important Attributes

xxx

### External Dependencies

xxx

### Elementary Functions

xxx


## Future Plans
### Core Features Not Yet Implemented and Plan to Add

We are still in the process of fully handling vectorization.  The core calculations in forward mode currently support computing the partial of a function $f$ from $\mathbb{R}^n$ to $\mathbb{R}$, but we have not yet handled the fully generalized case of function from $\mathbb{R}^n$ to $\mathbb{R}^m $.  

We can compute a forward mode differentiation for an arbitrary seed value.  We plan to add convenience methods that will return three quantities of common interest to end users:

  - `jacobian(X)`, the Jacobian of the function evaluated at inputs $X$
  - `hessian(X)`, the Hessian of the function evaluated at inputs $X$
  - `partial(var_name, X)`, the partial derivative of the function with respect to the named variable, when evaluated at inputs $X$
  - `Stack([fs])` would be a constructor for a Fluxion that mapped from $\mathbb{R^n}$ to $\mathbb{R^m}$.  It would take as input a Python list of Fluxion instances.  In the initial implementation, each function $f$ in the list would need to share the same input variables.  Eventually the function arguments could be merged.  The resulting function $F$ would have $F_i = fs[i]$, that is the scalar valued function $f_i$ would be the $i^{th}$ component of $F$ in $\mathbb{R^m}$.

We are having team discussions about whether to implement the remaining mathematical dunder methods: `__floordiv__`, `__mod__`,` __divmod__`, `__abs__`, `__round__`, `__trunc__`, `__floor__`, `__ceil__`.  Arguments in favor are that they are built-in methods that apply to numerical objects.  The main argument against is that they are not differentiable functions.

### Value Added Features

We plan to add the following "value added" features that extend off the core forward mode differentiation:

  - Implicit Time Integrator (ODE solver)
  - Root Finding with Newton's Method as well as several quasi-Newton methods
  - Optimization toolkit if time permits - basic convex optimization problems
  - Application that uses the implicit time integrator: simulation of the Solar System
      - Obtain the positions and velocities of all 8 planets in the Solar system as of a known date
      - Set up a symplectic integrator with sufficient accuracy to reliably forecast the movement of the planets over a multi-year time scale
      - Use the simulator to create a cool visual effect, either still images or a movie

### Possible Software Changes

The biggest area that needs work is the way we handle variable binding from different kinds of outputs.  That part of the code base is still a bit messy and confusing.

One challenge we've faced has to do with the array sizing conventions for accepting inputs and returning outputs.  In general, inputs will have size $T$ by $n$, where $T$ is the number of "sample points" and $n$ is the number of variables a function accepts as arguments.  Output function values will have size $T$ by $m$; function evaluations in forward mode with one seed value will also have size $T$ by $m$; and the Jacobian will have size $T$ by $m$ by $n$.  

We decided early on to infer the size of an input array when possible.  For example, if a fluxion `f` represents the function $f(x) = x sin(x)$, that is a function from $\mathbb{R}$ to $\mathbb{R}$.  If presented with input that is an array of shape (3,), the library treats this as having $T=3$ and $n=1$, since that is the unique legal input shape.  On the other hand, if a fluxion `g` represents the function $g(x, y, z) = x^3y^2z$, input of shape (3,) must be a single point, i.e. $T=1$ and $n=3$.  Similarly, if we return the output of the first calculation, it will have shape (1,) rather than shape (1,1).

While this is a convenient convention for end users consuming the functions, it has led to some obscure code where we are trying to infer the tensor ranks and shapes of Fluxions in multiple places.  We have a proposal to streamline this by having the calculations done with implementation methods that always take as inputs and outputs arrays with all indices, even if they are 1.  All function inputs would have shapes `(T, n)`, all value outputs shape `(T, m)`, forward_mode output with one seed value would also have shape `(T, m)`, and the Jacobian would have shape `(T, m, n)`.  For example, the Fluxion class would do this in a val_impl() method.  Then the public interface val() would dispatch a call to val_impl(), and apply np.squeeze() to the results.

Another pending change has to due with the composition of Fluxions that include elementary functions.  This is not currently working the way we want it to.  The cause of the issue stems from trying to make `fl.sin(np.vector[0, 1, 2])` return values for immediate use.  While you might want to see 0.0 or (0.0, 1.0) on the screen when you call `fl.sin(0)`, that setup has interfered with compositions when an elementary function is on an inner node.  We have a pending proposal to address this.  To see immediate answers on the screen, a user would need to type `fl.sin(0.val())`.  `fl.sin(0)` would return the somewhat obscure looking `fl(Const(0))`, but that would make elementary function nodes fully composable.

### Primary Challenges

The main challenges in writing this package so far have been very different from the main concepts that drive differentiation in forward mode.  This has been surprising to us.  The core mathematics including the chain rule, product and quotient rules, and the derivatives of elementary functions have been a breeze.  The messiest part of our code has been dealing with the various book-keeping of parsing inputs in different formats.  At the very start we opted to allow users to provide inputs in different formats including numpy arrays, dictionaries with variable values bound to their names, and lists of values passed by position.  This will make the end product more convenient to use, but has led to complexities in the argument parsing and variable binding sections of the code.

A second challenge has been vectorization.  Getting things up and running quickly for functions from $\mathbb{R}$ to $\mathbb{R}$ proved much simpler than adding support for vectorizing for inputs in $\mathbb{R^n}$.  One source of confusion for us was that for functions with only one input, there is no real need to specify a seed value, because the only seed of interest is effectively $dx=1$ which corresponds to $\partial f / \partial x = f'(x)$.  This meant that we prototyped some of our code base with an API where we didn't clearly pass in a seed value, but just used this default.  Eventually we figured this out and updated our APIs and logic.

Learning to use new tools effectively has been rewarding but also at times tough.  Everyone on the team has had a solid amount of experience writing Python code at this point.  We've been using Git all semester, but using Git to version control your files and synchronize between your laptop and your desktop is much, much easier than using Git to collaborate with other people!  In a similar vein, we've been learning to use tools like pytest and coverage testing on small assignments we write on our own.  Getting an automated testing suite working on Travis with coverage testing on Coveralls has been worthwhile but required an effort.  These tools also led us to a slew of challenges relating to module and package imports.  We had a module and package configuration that worked fine for all of us on our local machines, but when we tried to get our tests to run on Travis we were hitting a number of confusing errors in import statements.  Finally we ended up solving this issue, including with some slightly obscure code that runs import statements differently depending on whether a module is being run as an import or run as `__main__`.  Looking at it now, this seems clear, but at the time it was confusing and consumed many hours and Stack Overflow searches.

Perhaps the most substantive challenge has been learning to work together efficiently.  We are fortunate in that everyone on the team has been working hard, getting along, and contributing.  But we've had multiple situations where one person wrote code that another person on the team had trouble understanding when their work tasks overlapped.  Two examples of this were the overlap between name binding / argument parsing and vectorization, and elementary functions and testing.  With the benefit of hindsight, our team probably would have benefited if we had been more deliberate about first writing down an API and test cases before we implemented anything.  The optimal level of advanced planning and process deliberation on a team of 4 has proven to be much higher than on a solo project or pair effort.