from math import isclose
from discrete_probability.Dice import D6


def test_two_d6():
    actual = D6 + D6
    assert isclose(actual[2], 1 / 36)
    assert isclose(actual[7], 1 / 6)
    assert isclose(actual.mean(), 7)
