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
    def __init__(self, in_dim, out_shape, is_circular = False):
        """ Constructor of the Self Organizing Map."""
        self.in_dim = in_dim
        self.out_shape = out_shape
        self.is_circular = is_circular
        self.W = np.zeros((self.in_dim, self.out_shape[0], self.out_shape[1]))

    def initialize_weights(self,W) :
        # Initialize weight vector
        self.W = W

    def find_winner(self,x) :
        # Calculate distance of input array to every output weight
        dists = np.linalg.norm(self.W-np.expand_dims(x,axis=1),axis=0)
        # Find winner node
        winner_index = np.argmin(dists)
        winner_index_x = winner_index%np.shape(dists)[0]
        winner_index_y = int(winner_index/np.shape(dists)[1]) if np.shape(dists)[1] > 1 else 0
        return winner_index_x, winner_index_y

    def find_nearest_neighbors(self,winner_index_x,winner_index_y,th) :
        # Return nearest neighbors according to Manhattan distance
        neighbors_x, neighbors_y = [], []
        th_x = min(self.out_shape[0],th)
        th_y = min(self.out_shape[1],th)
        for ind_x in [winner_index_x-th_x + i for i in range(2*th_x+1)] :
            for ind_y in [winner_index_y-th_y + i for i in range(2*th_y+1)] :
                if (abs(winner_index_x-ind_x) + abs(winner_index_y-ind_y)) <= th :
                    if self.is_circular :
                        if (ind_x < 0) :
                            ind_x = ind_x + self.out_shape[0]
                        elif (ind_x >= self.out_shape[0]) :
                            ind_x = ind_x - self.out_shape[0]
                        if (ind_y < 0) :
                            ind_y = ind_y + self.out_shape[1]
                        elif (ind_y >= self.out_shape[1]) :
                            ind_y = ind_y - self.out_shape[1]
                        neighbors_x.append(ind_x)
                        neighbors_y.append(ind_y)
                    else :
                        if not ((ind_x < 0) or (ind_x >= self.out_shape[0]) or 
                                (ind_y < 0) or (ind_y >= self.out_shape[1])) :
                            neighbors_x.append(ind_x)
                            neighbors_y.append(ind_y)

        return neighbors_x, neighbors_y

    def update_weights(self,lr,x,neighbors_x,neighbors_y):
        # Update weights of closest neighbors
        n_neighbors = len(neighbors_x)
        updated_neighbors = []
        for i in range(n_neighbors) :
            if (neighbors_x[i],neighbors_y[i]) not in updated_neighbors :
                delta_weight = lr*(x-self.W[:,[neighbors_x[i]],[neighbors_y[i]]])
                self.W[:,[neighbors_x[i]],[neighbors_y[i]]] += delta_weight
                updated_neighbors.append((neighbors_x[i],neighbors_y[i]))

    def train(self,X,lr,n_epochs,init_th) :
        # In each epoch loop over all input patterns 
        N = np.shape(X)[1]
        th_step = init_th/n_epochs
        for e in range(n_epochs) :
            # Update neighbourhood th
            th = max(int(init_th-e*th_step),1)
            for i in range(N) :
                # Retrieve pattern
                x = X[:,[i]]
                # Find winner
                winner_index_x, winner_index_y = self.find_winner(x)
                # Find closest neighbors
                neighbors_x, neighbors_y = self.find_nearest_neighbors(winner_index_x,winner_index_y,th)
                # Update weights
                self.update_weights(lr,x,neighbors_x,neighbors_y)

    def output_topology(self,X) :
        # Loop over all input patterns 
        N = np.shape(X)[1]
        winners = []
        for i in range(N) :
            # Retrieve pattern
            x = X[:,[i]]
            # Find winner
            winner_index_x, winner_index_y = self.find_winner(x)
            # Append winner index
            winners.append((winner_index_x,winner_index_y))
        
        return winners
        
        
# HELPER FUNCTIONS
def plot_cities(X,names) :
    fig, ax = plt.subplots()
    ax.scatter(X[0,:],X[1,:], c=colors[0], label='KTH Buildings')
    for i, name in enumerate(names):
        ax.annotate(name, (X[0,i]+0.01, X[1,i]+0.01))
    ax.grid(visible = True)
    ax.legend()
    ax.set_title('Traveling KTH student')
    plt.show()

def plot_cities_route(X,path,names) :
    fig, ax = plt.subplots()
    path = np.append(path,path[:,[0]],axis=1)
    ax.scatter(X[0,:],X[1,:], c=colors[0], label='KTH Buildings')
    ax.plot(path[0,:],path[1,:],c=colors[1],label='Student route')
    for i, name in enumerate(names):
        ax.annotate(name, (X[0,i]+0.01, X[1,i]+0.01))
    ax.grid(visible = True)
    ax.legend()
    ax.set_title('Traveling KTH student')
    plt.show()

def calculate_perimeter(path) :
    path = np.append(path,path[:,[0]],axis=1)
    path_length = 0
    for i in range(np.shape(path)[1]-1) :
        path_length += np.sqrt((path[0,i]-path[0,i+1])**2 + (path[1,i]-path[1,i+1])**2)
    return path_length




