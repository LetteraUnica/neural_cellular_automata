from typing import Tuple

import numpy as np


class CachedDiscreteIntegrator:
    """Integrates a function between two positive integer values and keeps the integral cached for faster evaluation"""

    def __init__(self, function):
        self.function = function
        self.integral = []

    @staticmethod
    def swap(a: int, b: int) -> Tuple[int, int]:
        """Returns the two elements swapped"""
        c = a
        a = b
        b = c

        return a, b

    def last_value(self):
        """Returns the last value in the integral"""
        if len(self.integral) > 0:
            return self.integral[-1]
        else:
            return 0

    def integrate(self, a: int, b: int) -> float:
        """Integrates the function between a and b"""
        if a > b:
            a, b = self.swap(a, b)

        if a < 0:
            return 0

        if len(self.integral) < b:
            new_function_values = [self.function(i) for i in range(len(self.integral), b+1)]

            for value in new_function_values:
                self.integral.append(value + self.last_value())

        return self.integral[b] - self.integral[a]