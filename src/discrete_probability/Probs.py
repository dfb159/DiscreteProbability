from decimal import Decimal
from fractions import Fraction
from functools import reduce
from itertools import product
from collections import defaultdict
from math import prod
from operator import add, mul, sub
from typing import (
    Callable,
    Concatenate,
    Iterable,
    Self,
    Tuple,
)
import gmpy2


type BinaryPrecision = gmpy2.mpz | gmpy2.mpfr
type Number[
    P: (
        float,
        Decimal,
        Fraction,
    )
] = int | P
type ValuePair[
    P: (
        float,
        Decimal,
        Fraction,
    )
] = Tuple[Number[P], Number[P]]
type Merged[
    P: (
        float,
        Decimal,
        Fraction,
    )
] = Number[
    P
] | Probs[P]


def _wrap_number[P: (
    float,
    Decimal,
    Fraction,
)](value: Merged[P]) -> "Probs[P]":
    if isinstance(value, (int, float, Decimal, Fraction)):
        return const(value)
    return value


def wrap_number[P: (
    float,
    Decimal,
    Fraction,
), R](func: Callable[Concatenate["Probs[P]", ...], R]) -> Callable[Concatenate[Merged[P], ...], R]:
    def wrapper(*values: Merged[P]):
        converted = [_wrap_number(v) for v in values]
        return func(*converted)

    return wrapper


def wrap_object[P: (
    float,
    Decimal,
    Fraction,
), R](default):
    def decorator(func: Callable[Concatenate[Probs[P], ...], Number[P]]) -> Callable[Concatenate[Probs[P] | object, ...], Number[P] | R]:
        def wrapper(*values: Merged[P] | object):
            if all(map(lambda v: isinstance(v, (int, float, Decimal, Fraction, Probs)), values)):
                return func(*values)  # type: ignore
            return default

        return wrapper

    return decorator


class Probs[P: (float, Decimal, Fraction)]:
    """A discrete probability distribution. Holds the probabilities for a set of values."""

    values: dict[Number[P], Number[P]]

    def __init__(self, values: Iterable[ValuePair[P]]):
        """Initializes a new distribution from the given value pairs.

        Args:
            values (Iterable[Tuple[K, P]]): Iterator with all value pairs.
        """
        self.values = defaultdict()
        for v, p in values:
            self.values[v] += p

    def __contains__(self, element: Number[P]) -> bool:
        return element in self.values

    def __getitem__(self, index: Number[P]) -> Number[P] | None:
        if index not in self.values:
            return None
        return self.values[index]

    def __iter__(self):
        for x in self.values.items():
            yield x

    def mean(self):
        return sum(k * v for k, v in self)

    def __repr__(self):
        return f"Probs::{repr(dict(self.values))}"

    def __str__(self):
        return f"Probs::{str(dict(self.values))}"

    def __len__(self):
        return len(self.values)

    def __abs__(self):
        return function(lambda x: abs(x), self)

    def __pos__(self):
        return function(lambda a: +a, self)

    def __neg__(self):
        return function(lambda a: -a, self)

    def __round__(self, ndigits: int = 0):
        return function(lambda a: round(a, ndigits), self)

    @wrap_object(False)
    def equal(self, other: Self) -> bool:
        """True, if the underlying values are the same. Value compare. Use ' self is other' for object compare."""
        return self.values == other.values

    @wrap_object(False)
    @wrap_number
    def __eq__(self, other: Self):  # type: ignore
        return predicate(lambda a, b: a == b, self, other)

    @wrap_object(False)
    @wrap_number
    def __ne__(self, other: Self) -> Number[P]:  # type: ignore
        return predicate(lambda a, b: a != b, self, other)

    @wrap_number
    def __add__(self, other: Self):
        return function(add, self, other)

    @wrap_number
    def __sub__(self, other: Self):
        return function(sub, self, other)

    @wrap_number
    def __mul__(self, other: Self):
        return function(mul, self, other)

    @wrap_number
    def __divmod__(self, modulo: Self):
        return function(lambda a, b: divmod(a, b), self, modulo)

    @wrap_number
    def __floordiv__(self, other: Self):
        return function(lambda a, b: a // b, self, other)

    @wrap_number
    def __truediv__(self, other: Self):
        return function(lambda a, b: a / b, self, other)

    @wrap_number
    def __mod__(self, other: Self):
        return function(lambda a, b: a % b, self, other)

    @wrap_number
    def __pow__(self, other: Self):
        return function(lambda a, b: a**b, self, other)

    @wrap_number
    def __ge__(self, other: Self):
        return predicate(lambda a, b: a >= b, self, other)

    @wrap_number
    def __gt__(self, other: Self):
        return predicate(lambda a, b: a > b, self, other)

    @wrap_number
    def __le__(self, other: Self):
        return predicate(lambda a, b: a <= b, self, other)

    @wrap_number
    def __lt__(self, other: Self):
        return predicate(lambda a, b: a <= b, self, other)


def uniform(to: int, fr: int = 0) -> Probs[float]:
    """Generates a uniform probability distribution between the given arguments.

    Args:
        to (int): End of range, exclusively
        fr (int, optional): Start of range. Defaults to 0.

    Returns:
        Probs: Uniformly spread discrete probability distribution.
    """
    p = 1 / (to - fr)
    return Probs([(n, p) for n in range(fr, to)])


def const[P: (
    float,
    Decimal,
    Fraction,
)](value: Number[P]) -> Probs[P]:
    """Generates a distribution with a single value.

    Args:
        value (ArithmeticNumber): The value to be used.

    Returns:
        Probs: Discrete probability distribution with a single value of probability one.
    """
    return Probs([(value, 1)])


def function[P: (
    float,
    Decimal,
    Fraction,
)](f: Callable[Concatenate[Number[P], ...], Number[P]], *props: Probs[P]) -> Probs[P]:
    """Evaluates a function and collects the result of all possible combinations of the input distributions.

    Args:
        f (Callable[Concatenate[Number[P], Number[P]]): The function to be evaluated.
        *p (Probs): The distributions to be inserted into the function.

    Returns:
        Probs: The resulting discrete probability distribution.

    Example:
    - calculate the probabilities of "throwing with advantage" in D&D
        >>> advantage = function(max, Dice.D20, Dice.D20)
    """

    def map_function(*state: Tuple[Number[P], Number[P]]) -> Tuple[Number[P], Number[P]]:
        v, p = zip(*state)  # switch the order of the arguments.
        value = f(*v)  # key of the resulting entry
        prob = prod(p)  # probability of the resulting entry
        return value, prob

    return Probs(map_function(*state) for state in product(*props))


def predicate[P: (
    float,
    Decimal,
    Fraction,
)](f: Callable[Concatenate[Number[P], ...], bool], *probs: Probs[P]) -> Number[P]:
    """Evaluates a boolean expression of all possible combinations of the input distributions.

    Args:
        f (Callable[..., bool]): The predicate to be evaluated.
        *p (Probs): The distributions to be inserted into the function.

    Returns:
        Probs: The resulting probability of the predicate over all input distributions.

    Example:
    - calculate the probability to throw better than the enemy
        >>> a = Dice.D20 + 3
        b = Dice.D20 - 1
        better = predicate(lambda x, y: x > y, a, b)
    """

    def map_predicate(*state: Tuple[Number[P], Number[P]]) -> Number[P]:
        v, p = zip(*state)  # switch the order of the arguments.
        return prod(p) if f(*v) else 0  # probability if valid, else zero

    return sum(map_predicate(*state) for state in product(*probs))
