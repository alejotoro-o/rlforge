import numpy as np

class AdamOptimizer:
    """
    Adam Optimizer implementation for neural networks.

    This class implements the Adam optimization algorithm for updating
    weights and biases in a feedforward neural network. It maintains
    exponential moving averages of gradients and squared gradients,
    applies bias correction, and updates parameters accordingly.

    Parameters
    ----------
    layer_dims : list of int
        Dimensions of each layer in the network. For example, [input_dim, hidden_dim, output_dim].
    learning_rate : float, optional
        Step size for parameter updates (default=1e-3).
    beta_m : float, optional
        Exponential decay rate for the first moment estimates (default=0.99).
    beta_v : float, optional
        Exponential decay rate for the second moment estimates (default=0.999).
    epsilon : float, optional
        Small constant added for numerical stability (default=1e-8).

    Attributes
    ----------
    m : list of dict
        First moment estimates (per layer, for weights "W" and biases "b").
    v : list of dict
        Second moment estimates (per layer, for weights "W" and biases "b").
    beta_m_product : float
        Cumulative product of beta_m for bias correction.
    beta_v_product : float
        Cumulative product of beta_v for bias correction.
    """

    def __init__(self, layer_dims, learning_rate=1e-3, beta_m=0.99, beta_v=0.999, epsilon=1e-8):
        """
        Initialize the Adam optimizer.

        Sets up moment estimates (m, v) for each layer parameter and
        initializes bias correction terms.

        Parameters
        ----------
        layer_dims : list of int
            Dimensions of each layer in the network.
        learning_rate : float, optional
            Step size for parameter updates (default=1e-3).
        beta_m : float, optional
            Exponential decay rate for the first moment estimates (default=0.99).
        beta_v : float, optional
            Exponential decay rate for the second moment estimates (default=0.999).
        epsilon : float, optional
            Small constant added for numerical stability (default=1e-8).
        """

        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.beta_m = beta_m
        self.beta_v = beta_v
        self.epsilon = epsilon

        self.m = [dict() for i in range(len(self.layer_dims) - 1)]
        self.v = [dict() for i in range(len(self.layer_dims) - 1)]

        for i in range(len(self.layer_dims) - 1):
            self.m[i]["W"] = np.zeros((self.layer_dims[i],self.layer_dims[i + 1]))
            self.m[i]["b"] = np.zeros((1,self.layer_dims[i + 1]))
            self.v[i]["W"] = np.zeros((self.layer_dims[i],self.layer_dims[i + 1]))
            self.v[i]["b"] = np.zeros((1,self.layer_dims[i + 1]))

        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    def update_weights(self, weights, grads):
        """
        Update network weights using Adam optimization.

        Applies moment updates, bias correction, and parameter updates
        for each layer in the network.

        Parameters
        ----------
        weights : list of dict
            Current weights and biases of the network. Each element corresponds
            to a layer and contains keys "W" and "b".
        grads : list of dict
            Gradients of the weights and biases for each layer. Each element
            corresponds to a layer and contains keys "W" and "b".

        Returns
        -------
        list of dict
            Updated weights and biases after applying Adam optimization.

        Notes
        -----
        - First moment (m) tracks the mean of gradients.
        - Second moment (v) tracks the uncentered variance of gradients.
        - Bias correction is applied to both m and v before updates.
        - Updates are performed in-place on the provided weights.
        """
    
        for i in range(len(self.layer_dims) - 1):
            for param in weights[i].keys():
        
                self.m[i][param] = self.beta_m*self.m[i][param] + (1 - self.beta_m)*grads[i][param]
                self.v[i][param] = self.beta_v*self.v[i][param] + (1 - self.beta_v)*(grads[i][param]**2)
                
                m_hat = self.m[i][param]/(1 - self.beta_m_product)
                v_hat = self.v[i][param]/(1 - self.beta_v_product) 
        
                weights[i][param] += self.learning_rate*m_hat/(np.sqrt(v_hat) + self.epsilon)

        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights