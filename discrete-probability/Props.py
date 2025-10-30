from itertools import product
from collections import defaultdict
from math import prod


class Probs:

    def _init_(self, values: list[(float, float)]):
        self.values = defaultdict(int)
        for v, p in values:
            self.values[v] += p

    def uniform(to, fr=0):
        p = 1 / (to - fr)
        return Probs([(n, p) for n in range(fr, to)])

    def const(c):
        return Probs([(c, 1)])

    def __iter__(self):
        for x in self.values.items():
            yield x

    def _repr_(self):
        return f"Probs::{repr(dict(self.values))}"

    def _str_(self):
        return f"Probs::{str(dict(self.values))}"

    def _len_(self):
        return len(self.values)

    def call(f, *p):
        return Probs((f(*n), prod(p)) for n, p in map(lambda x: zip(*x), product(*p)))

    def _add_(self, other):
        return Probs.call(lambda a, b: a + b, self, other)
