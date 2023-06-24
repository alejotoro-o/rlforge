import numpy as np

class LinearRegression:

    def __init__(self, input_dim, output_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.zeros((self.output_dim, self.input_dim))     

    def predict(self, x, tile_coding_indices=False):

        if tile_coding_indices:
            out = np.sum(self.weights[:,x], axis=1)
        else:
            x = x.reshape((-1,1))
            out = np.dot(self.weights,x)

        return out
    
    def update_weights(self, step_size , gradients, action=0, state=[], tile_coding_indices=False):

        if tile_coding_indices:
            self.weights[action, state] += step_size*gradients
        else:
            self.weights[action,:] += step_size*gradients

    def reset_weights(self):

        self.weights = np.zeros((self.output_dim, self.input_dim)) 