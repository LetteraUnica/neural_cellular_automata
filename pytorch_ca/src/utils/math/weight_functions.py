import numpy as np
class ConstantWeight:
    """Returns 1 if the iteration is in the interval [initial_step, end_step], 0 otherwise."""

    def __init__(self, initial_step, end_step=np.inf,constant=1):
        self.initial_step = initial_step
        self.end_step = end_step
        self.constant = constant

    def __call__(self, current_iteration, **kwargs):
        return self.constant if self.initial_step <= current_iteration <= self.end_step else 0


class NormalizedSigmoid:
    def __init__(self,sigma,x_0):
        self.sigma = sigma
        self.x_0 = x_0

    def sigmoid(self,x):
        x=(x-self.x_0)/self.sigma
        return 1/(1+np.exp(-x))

    def softplus(self,x):
        x=(x-self.x_0)/self.sigma
        return np.log(1+np.exp(x))*self.sigma

    def __call__(self,current_iteration,start_iteration,end_iteration,*args,**kwargs):
        N=self.softplus(end_iteration)-self.softplus(start_iteration)+1e-8
        return self.sigmoid(current_iteration)/N


class NormalizedExponential:
    def __init__(self,tau=1,x_0=0):
        self.tau = tau
        self.x_0 = x_0

    def exp(self,x):
        x=(x-self.x_0)/self.tau
        return np.exp(x)


    def __call__(self,current_iteration,start_iteration,end_iteration,*args,**kwargs):
        N=(self.exp(end_iteration)-self.exp(start_iteration))*self.tau+1e-8
        return self.exp(current_iteration)/N