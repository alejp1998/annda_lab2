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

# RADIAL BASIS FUNCTIONS
class RBF:
    def __init__(self, arch, rad_basis_fn, lr, n_epochs):
        """ Constructor of the RBF Network."""
        self.arch = arch
        self.n = arch[1]
        self.n_layers = 2
        self.lr = lr
        self.n_epochs = n_epochs
        self.W = np.zeros((self.arch[1], self.arch[2]))
        self.mus = np.zeros((self.arch[1],arch[0]))
        self.sigmas = np.ones((self.arch[1],1))
        self.rad_basis_fn = rad_basis_fn

    def initialize_weights(self) :
        # Initialize weights
        self.W = np.random.normal(0, 1, size=(self.arch[1], self.arch[2]))

    def initialize_rad_basis_fns(self,mus,sigmas) :
        # Initialize RBFs characteristics
        self.mus = mus
        self.sigmas = sigmas

    def rad_basis_mat(self,X) :
        # Number of input patterns
        N = np.shape(X)[1]
        out_dim = np.shape(X)[0]
        # Compute RBF matrix
        Phi = np.zeros((N,self.n))
        for i in range(self.n) :
            # Vectorize radial basis transfer function
            def rad_bas_fn_i(x) :
                return self.rad_basis_fn(x,self.mus[[i],:],self.sigmas[i,:])
            if out_dim > 1 :
                Phi[:,i] = np.apply_along_axis(rad_bas_fn_i, 0, X)
            else :
                vfunc = np.vectorize(rad_bas_fn_i)
                Phi[:,i] = vfunc(X)
        self.Phi = Phi
        return Phi

    def winner_takes_all(self,X,lr,unique_winner=True) :
        # Shuffle data points
        N = np.shape(X)[1]
        shuffled_indices = np.random.choice(N, size=N, replace=False)
        X = X[:,shuffled_indices]
        # Iterate over patterns (columns)
        for i in range(N) :
            # Find closest RBF node
            distances = X[:,[i]].T - self.mus
            distances_norm = np.linalg.norm(distances,axis=1)
            # Update center of closest node
            if unique_winner :
                min_i = np.argmin(distances_norm)
                self.mus[min_i,:] = self.mus[min_i,:] + lr*distances[min_i,:]
            else :
                min_indices = np.argsort(distances_norm)
                power = 0
                n_mins = 3 if len(distances_norm) >= 3 else 1
                for min_i in min_indices[:n_mins] :
                    min_i = np.argmin(distances_norm)
                    self.mus[min_i,:] = self.mus[min_i,:] + (lr**power)*distances[min_i,:]
                    power += 1
                
    def forward_pass(self,X) :
        # Compute RBF matrix
        Phi = self.rad_basis_mat(X)
        # Compute output
        f = Phi @ self.W
        return f
    
    def least_squares(self,X,f) :
        # Compute RBF matrix
        Phi = self.rad_basis_mat(X)
        # Train weights with LS method
        temp1 = np.linalg.inv(Phi.T @ Phi)
        temp2 = Phi.T @ f.T
        self.W = temp1 @ temp2

    def delta_rule(self,X_train,f_train,X_test,f_test,n_epochs) :
        # Learning curves
        res_error_epochs = []
        res_error_epochs_test = []
        N = np.shape(X_train)[1]
        # Iterate over epochs
        for epoch in range(n_epochs) :
            random_col_indices = np.random.choice(N, size=N, replace=False)
            X_train = X_train[:,random_col_indices]
            f_train = f_train[:,random_col_indices]
            for i in range(N) :
                # Compute RBF vector
                Phi = self.rad_basis_mat(X_train[:,[i]])
                # Train weights with Delta Rule method
                error = f_train[:,[i]].T - (Phi @ self.W)
                delta_weights = self.lr*(Phi.T @ error)
                self.W = self.W + delta_weights
            # Update training and test residual error
            res_error_epochs.append(residual_error(f_train,self.forward_pass(X_train).T))
            res_error_epochs_test.append(residual_error(f_test,self.forward_pass(X_test).T))

        return res_error_epochs, res_error_epochs_test


# HELPER FUNCTIONS
def gaussian_tf (x,mu,sigma):
    return np.exp(-np.linalg.norm(x-mu)**2/(2*sigma**2))

def sin(x,variance) : 
    return np.sin(2*x) + np.random.normal(0, variance)

def square(x,variance) :
    return 2*(np.sin(2*x)>=0) - 1 + np.random.normal(0, variance) 

def sample_f(X,fn,variance) :
    def fn_var (x) :
        return fn(x,variance)
    vfunc = np.vectorize(fn_var)
    return vfunc(X)

def plot_data(X_train,X_test,f_train,f_test) :
    fig, ax = plt.subplots()
    ax.grid(visible = True)
    ax.set_title('Training and test data')
    ax.scatter(X_train,f_train,label='Train',color=colors[0])
    ax.scatter(X_test,f_test,label='Test',color=colors[1])
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.show()

def plot_series(rbfnet,X,series,series_pred,title) :
    fig, ax = plt.subplots()
    ax.grid(visible = True)
    ax.set_title(title)
    ax.plot(X,series,color=colors[0],label='True')
    ax.plot(X,series_pred,color=colors[1],label='Predicted')
    ax.scatter(rbfnet.mus,np.zeros(np.shape(rbfnet.mus)[0]),s=rbfnet.sigmas*100,color='green',marker='x',label='RBF nodes')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.show()

def plot_error(errors,errors_test,title) :
    fig, ax = plt.subplots()
    ax.grid(visible = True)
    ax.set_title(title+' over epochs')
    ax.plot(errors,label='Training')
    ax.plot(errors_test,label='Test')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    plt.show()

def plot_input_space(rbfnet,X) :
    fig, ax = plt.subplots()
    ax.scatter(X[0,:],X[1,:], c=colors[0], label='Patterns')
    ax.scatter(rbfnet.mus.T[0,:],rbfnet.mus.T[1,:], s=rbfnet.sigmas*100, c='green', marker='x', label='RBF nodes')
    ax.grid(visible = True)
    ax.legend()
    ax.set_title('Input Space')
    plt.show()

def residual_error(f,f_pred) :
    return np.absolute(f-f_pred).sum()/np.shape(f_pred)[1]

        



