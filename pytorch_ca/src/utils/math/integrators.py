from typing import Sequence

from functools import cache

import numpy as np


class CachedSummer:
    """Integrates a function between two positive integer values and keeps the integral cached for faster evaluation"""

    def __init__(self, function):
        self.function = function
        self.sum_until = np.vectorize(self._sum_until)
        self.sum_between = np.vectorize(self._sum_between)

    @cache
    def _sum_until(self, n):
        """Integrates the function in [0, n]"""
        if n < 0:
            return 0
        return self._sum_until(n-1) + self.function(n)

    def _sum_between(self, a: int, b: int) -> float:
        """Integrates the function in [a, b]"""
        return self._sum_until(b) - self._sum_until(a-1)
