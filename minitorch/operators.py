"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """f(x, y) = x * y"""
    return x * y


def id(x: float) -> float:
    """f(x) = x"""
    return x


def add(x: float, y: float) -> float:
    """f(x, y) = x + y"""
    return x + y


def neg(x: float) -> float:
    """f(x) = -x"""
    if type(x) is not float:
        x = float(x)
    return -x


def lt(x: float, y: float) -> float:
    """f(x, y) = x < y"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """f(x, y) = x == y"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """f(x, y) = max(x, y)"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """f(x, y) = |x - y| < 1e-2"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """f(x) = 1.0 / (1.0 + exp(-x))"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """f(x) = max(0, x)"""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """f(x) = log(x)"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """f(x) = exp(x)"""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """f(x) = d / x"""
    return d / (x + EPS)


def relu_back(x: float, d: float) -> float:
    """f(x) = d if x > 0 else 0"""
    return d if x > 0 else 0.0


def inv(x: float) -> float:
    """f(x) = 1.0 / x"""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """f(x) = -1.0 / x^2"""
    return -1.0 / (x**2) * d


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher order function for mapping a function over a list."""

    def _map(lst: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in lst:
            ret.append(fn(x))
        return ret

    return _map


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate a list of floats."""
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher order function for element-wise function application."""

    def _zipWith(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(xs, ys):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def addLists(xs: Iterable[float], ys: Iterable[float]) -> Iterable[float]:
    """Element-wise addition of two lists."""
    return zipWith(add)(xs, ys)


def reduce(
    fn: Callable[[float, float], float], init: float
) -> Callable[[Iterable[float]], float]:
    """Higher order function for reducing a list with a function."""

    def _reduce(lst: Iterable[float]) -> float:
        ret = init
        for x in lst:
            ret = fn(ret, x)
        return ret

    return _reduce


def sum(xs: Iterable[float]) -> float:
    """Sum a list of floats."""
    return reduce(add, 0.0)(xs)


def prod(xs: Iterable[float]) -> float:
    """Product of a list of floats."""
    return reduce(mul, 1.0)(xs)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists
