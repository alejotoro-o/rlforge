import numpy as np

from ..optimizers import AdamOptimizer

class MLP:
    """
    A simple Multi-Layer Perceptron (MLP) implementation with support for
    forward propagation, backward propagation, and weight updates using
    optimizers such as Adam.

    This class builds a fully connected feedforward neural network with
    configurable hidden layers. It uses ReLU activations for hidden layers
    and supports Saxe orthogonal initialization for weights.

    Parameters
    ----------
    input_dim : int
        Dimension of the input features.
    output_dim : int
        Dimension of the output layer.
    hidden_layers : list of int, optional
        Sizes of hidden layers (default=[2]).
    learning_rate : float, optional
        Learning rate for optimizer (default=1e-3).
    optimizer : str or object, optional
        Optimizer to use. If "adam", initializes an AdamOptimizer internally.
        Otherwise, accepts a custom optimizer object with an `update_weights`
        method (default="adam").

    Attributes
    ----------
    layer_dims : list of int
        Dimensions of all layers including input, hidden, and output.
    weights : list of dict
        Network parameters. Each dict contains:
        - "W" : np.ndarray, weight matrix
        - "b" : np.ndarray, bias vector
    optimizer : object
        Optimizer instance used for weight updates.
    """

    def __init__(self, input_dim, output_dim, hidden_layers=[2], learning_rate=1e-3, optimizer="adam"):
        self.layer_dims = [input_dim] + hidden_layers + [output_dim]
        self.learning_rate = learning_rate
        self.initialize_weights()
        if optimizer == "adam":
            self.optimizer = AdamOptimizer(self.layer_dims, self.learning_rate)
        else:
            self.optimizer = optimizer

    def forward_propagation(self, state):
        """
        Perform forward propagation through the network.

        Applies linear transformations followed by ReLU activations for
        hidden layers. The output layer is linear.

        Parameters
        ----------
        state : np.ndarray, shape (batch_size, input_dim)
            Input data.

        Returns
        -------
        tuple
            output_values : np.ndarray, shape (batch_size, output_dim)
                Final output of the network.
            cache : list of dict
                Cached intermediate values (A, Z) for each layer, used in
                backward propagation.
        """

        cache = [dict() for i in range(len(self.layer_dims) - 1)]

        Z = 0  
        A = state

        for i in range(len(self.layer_dims) - 2):

            cache[i]["A"] = A
            cache[i]["Z"] = Z

            Z = np.dot(A,self.weights[i]["W"]) + self.weights[i]["b"]
            A = np.maximum(Z,0)

        cache[-1]["A"] = A
        cache[-1]["Z"] = Z

        output_values = np.dot(A, self.weights[-1]["W"]) + self.weights[-1]["b"] 

        return output_values, cache
    

    def backward_propagation(self, mini_batch_size, delta, cache):
        """
        Perform backward propagation to compute gradients.

        Uses cached activations and pre-activations to compute gradients
        for weights and biases with respect to the loss.

        Parameters
        ----------
        mini_batch_size : int
            Size of the mini-batch used for normalization.
        delta : np.ndarray
            Gradient of the loss with respect to the output layer.
        cache : list of dict
            Cached values from forward propagation.

        Returns
        -------
        list of dict
            Gradients for each layer. Each dict contains:
            - "W" : np.ndarray, gradient of weights
            - "b" : np.ndarray, gradient of biases
        """

        grads = [dict() for i in range(len(self.layer_dims) - 1)]

        dZ = delta
    
        grads[-1]["W"] = (1./mini_batch_size)*np.dot(cache[-1]["A"].T,dZ)
        grads[-1]["b"] = (1./mini_batch_size)*np.sum(dZ, axis=0, keepdims=True)

        for i in reversed(range(1,len(self.layer_dims) - 1)):

            dg = (cache[i]["Z"] > 0).astype(float)
            dZ = np.dot(dZ,self.weights[i]["W"].T)*dg

            grads[i-1]["W"] = (1./mini_batch_size)*np.dot(cache[i-1]["A"].T,dZ)
            grads[i-1]["b"] = (1./mini_batch_size)*np.sum(dZ, axis=0, keepdims=True)     

        return grads

    def update_weights(self, grads):
        """
        Update network weights using the optimizer.

        Parameters
        ----------
        grads : list of dict
            Gradients for each layer, containing "W" and "b".

        Returns
        -------
        None
            Updates weights in-place using the optimizer.
        """

        self.weights = self.optimizer.update_weights(self.weights, grads)

    # Initialize weights
    def initialize_weights(self):
        """
        Initialize network weights and biases.

        Uses Saxe orthogonal initialization for weights and zeros for biases.

        Returns
        -------
        None
            Initializes self.weights with random orthogonal matrices and zeros.
        """

        self.weights = [dict() for i in range(len(self.layer_dims) - 1)]
        for i in range(len(self.layer_dims) - 1):
            self.weights[i]["W"] = self.__saxe_init(self.layer_dims[i],self.layer_dims[i + 1])
            self.weights[i]["b"] = np.zeros((1,self.layer_dims[i + 1]))

    def __saxe_init(self,rows,cols):
        """
        Perform Saxe orthogonal initialization for a weight matrix.

        Generates a random Gaussian matrix, applies QR decomposition, and
        adjusts signs to ensure orthogonality.

        Parameters
        ----------
        rows : int
            Number of rows in the weight matrix.
        cols : int
            Number of columns in the weight matrix.

        Returns
        -------
        np.ndarray
            Orthogonally initialized weight matrix of shape (rows, cols).
        """

        tensor = np.random.normal(0,1,(rows,cols))
        if rows < cols:
            tensor = tensor.T
        tensor, r = np.linalg.qr(tensor)
        d = np.diag(r, 0)
        ph = np.sign(d)
        tensor *= ph

        if rows < cols:
            tensor = tensor.T

        return tensor
    