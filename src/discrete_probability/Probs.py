from decimal import Decimal
from fractions import Fraction
from functools import reduce
from itertools import product
from collections import defaultdict
from math import prod
from operator import add, mul, sub
from typing import Callable, Iterable, Self, Tuple, TypeVar, Union, cast
import gmpy2

ArithmeticNumber = Union[int, float, Decimal, Fraction, gmpy2.mpz, gmpy2.mpfr]
Probability = float
ValuePair = Tuple[ArithmeticNumber, Probability]


Merged = Union[ArithmeticNumber, "Probs"]


def _wrap_number(value: Merged) -> "Probs":
    if isinstance(value, ArithmeticNumber):
        return const(value)
    else:
        return value


R = TypeVar("R")


def wrap_number(func: Callable[..., R]) -> Callable[..., R]:
    def wrapper(*values: Merged):
        converted = [_wrap_number(v) for v in values]
        return func(*converted)

    return wrapper


T = TypeVar("T")


def wrap_object(default: T) -> Callable[..., Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*values: Merged | object) -> T:
            if any(map(lambda v: not isinstance(v, Merged), values)):
                return default
            return func(*values)

        return wrapper

    return decorator


class Probs:
    """A discrete probability distribution. Holds the probabilities for a set of values."""

    values: dict[ArithmeticNumber, Probability]

    def __init__(self, values: Iterable[ValuePair]):
        """Initializes a new distribution from the given value pairs.

        Args:
            values (Iterable[ValuePair]): Iterator with all value pairs.
        """
        self.values = defaultdict(int)
        for v, p in values:
            self.values[v] += p

    def __contains__(self, element):
        return element in self.values

    def __iter__(self):
        for x in self.values.items():
            yield x

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
        """True, if the underlying values are the same. Value compare."""
        return self.values == other.values

    @wrap_object(False)
    @wrap_number
    def __eq__(self, other: Self) -> ArithmeticNumber:
        return predicate(lambda a, b: a == b, self, other)

    @wrap_object(False)
    @wrap_number
    def __ne__(self, other: Self) -> ArithmeticNumber:
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


def uniform(to: int, fr: int = 0) -> Probs:
    """Generates a uniform probability distribution between the given arguments.

    Args:
        to (int): End of range, exclusively
        fr (int, optional): Start of range. Defaults to 0.

    Returns:
        Probs: Uniformly spread discrete probability distribution.
    """
    p = 1 / (to - fr)
    return Probs([(n, p) for n in range(fr, to)])


def const(value: ArithmeticNumber) -> Probs:
    """Generates a distribution with a single value.

    Args:
        value (ArithmeticNumber): The value to be used.

    Returns:
        Probs: Discrete probability distribution with a single value of probability one.
    """
    return Probs([(value, 1)])


def function(f: Callable[..., ArithmeticNumber], *p: Probs) -> Probs:
    """Evaluates a function and collects the result of all possible combinations of the input distributions.

    Args:
        f (Callable[..., ArithmeticNumber]): The function to be evaluated.
        *p (Probs): The distributions to be inserted into the function.

    Returns:
        Probs: The resulting discrete probability distribution.

    Example:
    - calculate the probabilities of "throwing with advantage" in D&D
        >>> advantage = function(max, Dice.D20, Dice.D20)
    """
    return Probs((f(*n), prod(p)) for n, p in map(lambda x: zip(*x), product(*p)))


def predicate(f: Callable[..., bool], *probs: Probs) -> ArithmeticNumber:
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
    return sum(prod(p) for n, p in map(lambda x: zip(*x), product(*probs)) if f(*n))  # type: ignore
