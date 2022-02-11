# -*- coding: utf-8 -*-

# TEAM MEMBERS

# Alejandro Jarabo Peñas
# 19980430-T472
# aljp@kth.se

# Miguel Garcia Naude
# 19980512-T697
# magn2@kth.se

# Jonne van Haastregt
# 20010713-3316 
# jmvh@kth.se 

# Load packages
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary variables
colors = ['#1E90FF','#FF69B4']

# SELF ORGANIZING MAP
class SOM:
    def __init__(self, arch, activation_fn, activation_fn_der, momentum, alpha, lr, n_epochs):
        """ Constructor of the Neural Network."""
        self.arch = arch
        self.n_layers = len(self.arch)-1
        self.momentum = momentum
        self.alpha = alpha
        self.lr = lr
        self.n_epochs = n_epochs
        self.weight_matrices = [np.zeros((self.arch[layer+1], self.arch[layer]+1)) for layer in range(self.n_layers)]
        self.momentum_matrices = self.weight_matrices
        self.activation_fn = activation_fn
        self.activation_fn_der = activation_fn_der
        self.forward_mem = [i for i in range(self.n_layers)]
        self.backward_mem = [i for i in range(self.n_layers)]
        self.X = 0
        self.T = 0
        self.O = 0

        
# HELPER FUNCTIONS
def activation_fn (x):
    return 2/(1 + np.exp(-x)) - 1

def activation_fn_der (x) :
    return (1/2)*(1 + activation_fn(x))*(1 - activation_fn(x))

def plot_data(X,T) :
    fig, ax = plt.subplots()
    ax.scatter(X[0,T>0],X[1,T>0], c=colors[0], label='Class B')
    ax.scatter(X[0,T<0],X[1,T<0], c=colors[1], label='Class A')
    ax.grid(visible = True)
    ax.legend()
    ax.set_title('Patterns and Labels')
    plt.show()



