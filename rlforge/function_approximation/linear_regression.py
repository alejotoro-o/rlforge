import numpy as np

class LinearRegression:
    """
    A simple linear regression model with support for tile coding.

    This class implements a linear function approximator for reinforcement
    learning or supervised learning tasks. It maintains a weight matrix
    and provides methods for prediction, weight updates, and reset.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features.
    output_dim : int
        Dimension of the output (e.g., number of actions or targets).

    Attributes
    ----------
    weights : np.ndarray, shape (output_dim, input_dim)
        Weight matrix initialized to zeros.
    """

    def __init__(self, input_dim, output_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.weights = np.zeros((self.output_dim, self.input_dim))     

    def predict(self, x, tile_coding_indices=False):
        """
        Predict output values given input features.

        Supports both standard linear regression and tile coding indexing.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector.
        tile_coding_indices : bool, optional
            If True, interprets `x` as tile coding indices and sums
            corresponding weights (default=False).

        Returns
        -------
        np.ndarray
            Predicted output values of shape (output_dim,).
        """

        if tile_coding_indices:
            out = np.sum(self.weights[:,x], axis=1)
        else:
            x = x.reshape((-1,1))
            out = np.dot(self.weights,x)

        return out
    
    def update_weights(self, step_size , gradients, action=0, state=[], tile_coding_indices=False):
        """
        Update model weights using gradients.

        Supports both standard updates and tile coding updates.

        Parameters
        ----------
        step_size : float
            Learning rate or step size for weight updates.
        gradients : np.ndarray
            Gradient values for the update.
        action : int, optional
            Index of the output/action to update (default=0).
        state : list or np.ndarray, optional
            State indices used for tile coding updates (default=[]).
        tile_coding_indices : bool, optional
            If True, updates weights using tile coding indices (default=False).

        Returns
        -------
        None
            Updates weights in-place.
        """

        if tile_coding_indices:
            self.weights[action, state] += step_size*gradients
        else:
            self.weights[action,:] += step_size*gradients

    def reset_weights(self):
        """
        Reset all weights to zero.

        Returns
        -------
        None
            Resets the weight matrix to zeros of shape (output_dim, input_dim).
        """

        self.weights = np.zeros((self.output_dim, self.input_dim)) 