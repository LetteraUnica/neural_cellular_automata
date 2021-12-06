import torch
import numpy as np


class combination_function_generator:

    def __init__(self, tau, img_interval , kill_interval, kill_multiplier, device="cpu", tau2=None, tau_kill=0, tau_decay=20,tau2_decay=None):
        self.img_interval=img_interval
        self.kill_interval=kill_interval
        self.kill_multiplier=kill_multiplier
        self.log_step=img_interval[-1]
        self.device=device

        self.tau=tau
        if self.tau2==None:
            self.tau2=2*tau
        if self.tau2_decay==None:
            self.tau2_decay=tau_decay
        self.tau_decay=tau_decay
        self.tau_kill=tau_kill

    def exponential(self, x, tau, interval):
        n_steps = x + interval[0]
        if n_steps >= interval[0] and n_steps <= interval[1]:
            return np.exp(tau*x)
        return 0



class combination_function_generator_virus(combination_function_generator):

    def __call__(self, n_steps,n_epoch=0):  
        kill_multiplier=self.kill_multiplier*np.exp(-n_epoch*self.tau_kill)
        tau=self.tau*np.exp(-n_epoch/self.tau_decay)
        tau2=self.tau2*np.exp(-n_epoch/self.tau2_decay)

        m1=self.exponential(n_steps-self.img_interval[1], tau, img_interval)
        m2=kill_multiplier*self.exponential(n_steps-self.kill_interval[1],tau2, img_interval)

        return torch.tensor([m1,m2],device=self.device).float()


class combination_function_generator_growing(combination_function_generator):

    def __call__(self, n_steps,n_epoch=0):
        tau=self.tau*np.exp(-n_epoch/self.tau_decay)
        m1=self.exponential(self.img_interval[1]-n_steps, tau, self.img_interval)
        
        m2=0
        kill_multiplier=self.kill_multiplier*np.exp(-n_epoch*self.tau_kill)
        if n_steps >= self.kill_interval[0] and n_steps<= self.kill_interval[1]:
            m2=kill_multiplier
        
        return torch.tensor([m1,m2],device=self.device).float()