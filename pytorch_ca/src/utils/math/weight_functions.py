from numpy import inf
class ConstantWeight:
    """Returns 1 if the iteration is in the interval [initial_step, end_step], 0 otherwise."""

    def __init__(self, initial_step, end_step=inf,constant=1):
        self.initial_step = initial_step
        self.end_step = end_step
        self.constant = constant

    def __call__(self, current_iteration, **kwargs):
        return self.constant if self.initial_step <= current_iteration <= self.end_step else 0