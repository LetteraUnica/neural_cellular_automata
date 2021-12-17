from typing import Sequence

from functools import cache


class CachedSummer:
    """Integrates a function between two positive integer values and keeps the integral cached for faster evaluation"""

    def __init__(self, function):
        self.function = function

    @cache
    def sum_until(self, n):
        if n < 0:
            return 0
        return self.sum_until(n-1) + self.function(n)

    def sum_between(self, a: int, b: int) -> float:
        """Integrates the function in [a, b]"""
        return self.sum_until(b) - self.sum_until(a-1)

    def sum_between_vectorized(self, a: Sequence[int], b:Sequence[int]) -> Sequence[float]:
        """Integrates the function between a and b with a and b sequences"""
        return [self.sum_between(ai, bi) for (ai, bi) in zip(a, b)]
