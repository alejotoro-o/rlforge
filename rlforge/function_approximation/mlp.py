import numpy as np

from rlforge.optimizers import AdamOptimizer

class MLP:
    
    def __init__(self, input_dim, output_dim, hidden_layers=[2], learning_rate=1e-3, optimizer="adam"):
        self.layer_dims = [input_dim] + hidden_layers + [output_dim]
        self.learning_rate = learning_rate
        self.initialize_weights()
        if optimizer == "adam":
            self.optimizer = AdamOptimizer(self.layer_dims, self.learning_rate)
        else:
            self.optimizer = optimizer

    def forward_propagation(self, state):

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
        self.weights = self.optimizer.update_weights(self.weights, grads)

    # Initialize weights
    def initialize_weights(self):
    
        self.weights = [dict() for i in range(len(self.layer_dims) - 1)]
        for i in range(len(self.layer_dims) - 1):
            self.weights[i]["W"] = self.__saxe_init(self.layer_dims[i],self.layer_dims[i + 1])
            self.weights[i]["b"] = np.zeros((1,self.layer_dims[i + 1]))

    # Saxe weight initialization
    def __saxe_init(self,rows,cols):

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
    