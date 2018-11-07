# Milestone 2

## Introduction
Differential calculus was invented as a formal branch of mathematics in the middle of the 17th century independently by Isaac Newton and Gottfried Leibniz.  Newton's orginal term for a derivative with respect to time was a "fluxion," which gave its name to his last book, *Method of Fluxions*, published posthumously. Newton used differential calculus to solve the problem of the motion of the planets around the sun, and it has proven to be an essential tool for the sciences ever since.  In the modern era, essentially all scientific calculations of interest are performed on computers.  There are many scenarios where we know how to compute a function of interest and would like to efficiently evaluate its derivative(s).  The canonical example is root finding.  If we know a function's derivative, we can iteratively improve from a starting guess until we find a root using a simple procedure, Newton's Method.  Many phenomena of interest in physics and other sciences can be described as differential equations (either ODEs or PDEs).  The ability to efficiently evaluate derivatives is crucial in numerically solving these systems.
In recent years, Machine Learning has been a hot research area.  Solving ML problems often hinges on the ability to train a neural network using some form of a gradient descent algorithm.  As the name suggests, this algorithm requires the ability to evaluate not just the function (here the output of the neural net) but also its first derivative, which in practice is typically the change in the error metric with respect to the parameter values.  These neural networks are massively complex functions that can have millions of parameters.  A procedure of numerically differentiating the network by shifting the value of each parameter has cost that scales linearly with the number of parameters.  Some form of automatic differentiation is vital to the practical use of neural nets.  One of the most successful machine learning libraries is TensorFlow by Google, which at its core is an enterprise class automatic differentiation library.

## How to use the fluxions package?

### How to install (for end users)?
Our package is available on Test PyPI. Before installing ensure you have a Python3 environment with numpy installed available.

If you are using a Mac you can setup an appropriate virtual environment in your desired directory as follows:

```console
pip3 install virtualenv
virtualenv -p python3 venv
source venv/bin/activate

pip3 install numpy
```

Once you have an appropriate environment set up, you can install the fluxions package with the following command:

```console
pip3 install --index-url https://test.pypi.org/simple/ fluxions
```
### How to install (for developers)?

Clone the git repository to a location of your choice. The repository is located [here.](https://github.com/CS207-Final-Project-Group-10/cs207-FinalProject)

Ensure you have a Python3 environment available. If you want to use a virtual environment then execute the following code in the cloned directory.

```console
pip3 install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
```

Finally install the requirements

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

## Background
Calculus gives us a few simple rules we can apply to compute the derivative of a function that is the result of a combination of two more simple functions.  Calculus rules include the sum, product, and quotient of two functions.  The most important calculus rule in Automatic Differentiation is the Chain Rule of calculus.  This states that the derivative of a composition is the product of two derivatives.
$$f'(u(x)) = f'(u(x) \cdot u'(x)$$
The chain rule works in multiple dimensions.  If $f$ is a function from $\mathbb{R}^n$ to $\mathbb{R}^m$, its derivative an $m$ x $n$ matric called the Jacobian.  The chain rule in the multidimensional case tells us to take the matrix product of an $m$ x $r$ matrix and an $r$ x $n$ matrix to compute the derivative.

The essential idea of Automatic Differentiation is that any computation performed by a computer will consist of a set of basic steps, e.g. function evaluations.  These may be strung together in a complex graph with multiple steps, but each step will be a basic operation.  For the evaluation of a mathematical function, each step will consist of either a "primitive" or built-in function (e.g. +, -, x, /, sqrt, log, exp, pow, etc.) or a composition of these primitives.  As long as we know how to evaluate both the function f(x) and its derivative f'(x) at each step, we can trace the calculation through the computation graph and also keep track of the first derivative.  This is the idea at the heart of Automatic Differentiation.

The forward mode of Automatic Differentiation takes 3 arguments, a function, $f(x)$, a vector at which to evaluate the function,  $x \in \mathbb{R}^n$, and a vector of 'seed' values for the derivatives, $dx\in \mathbb{R}^
n$. Following the standard order of operations, the innermost elementary function calls of $f$ are evaluated at x and dx. As above, as long as we know how to evaluate both the elementary function f(x) and its derivative f'(x) at each step, pass the results of each to enclosing functions and apply the chain rule to keep track of the first derivative. Once all function calls are complete, the algorithm returns the values $f(x)$ and $f'(x)$.

For example, if we wish to use the forward mode algorithm to differentiate $f(x) = e^{x_1}cos(2x_1 + x_2)$ at the point $x = (x_1,x_2) = (0,\pi / 2)$, and we were only interested in the derivative with respect to $x_1$ we would pass the function $f$, the vector $x$, and $dx = (1,0)$. Starting from the innermost functions, we evaluate the derivative and the value of $e^{x_1}$. The value is the scalar 1 and the derivative is a vector $(1,0)$. Moving on, we evaluate the * operator on $2x_1$, giving a value 0 and the vector (2,0). The + operator gives the value $\pi/2$ and the vector (2,0). Cosine gives the scalar 0 and the vector $(-sin(2), 0)$. Finally, the multiplication operator between $e$ and $cos$ gives $f(x) = 0$ and  $f'(x) = (-sin(2),0)$ which would be the output. Because we passed the seed value $dx = (1,0)$, $f'(x) $ has a zero in the place corresponing to $x_2$. The zero in the seed carries through to the final derivative, only returning the non-zero entries.

## Software organization
- High-level overview of how the software is organized.
    * Directory structure
    * Basic modules and what they do
    * Where do tests live?  How are they run?  How are they integrated?


## Implementation details
* Implementation details
  - Description of current implementation.  This section goes deeper than the high level software
    organization section.
    * Try to think about the following:
      - Core data structures
      - Core classes
      - Important attributes
      - External dependencies
      - Elementary functions


## Future Plans
  - What aspects have you not implemented yet?  What else do you plan on implementing?
  - How will your software change?  
  - What will be the primary challenges?
  
  Future additions to the Fluxions package include:
  - Implementing the remaining mathematical dunder methods: __floordiv__, __mod__, __divmod__, __abs__, __round__, __trunc__, __floor__, __ceil__

  - Generalizing our vectorized implementation to handle matrices, and possibly higher-dimensional tensors.

