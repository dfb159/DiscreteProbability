from discrete_probability import Probs
from discrete_probability.Dice import D6


def test_two_d6():
    actual = D6 + D6
    assert actual[2] == 1 / 36
    assert actual[7] == 1 / 6
    assert actual.mean == 7
