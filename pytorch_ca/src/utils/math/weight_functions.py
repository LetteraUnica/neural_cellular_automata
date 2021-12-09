class ConstantWeight:
    """Returns 1 if the iteration is in the interval [initial_step, end_step], 0 otherwise."""
    def __init__(self, initial_step, end_step):
        self.initial_step = initial_step
        self.end_step = end_step
    
    def __call__(self, evolution_iters):
        return 1 if self.initial_step <= evolution_iters <= self.end_step else 0