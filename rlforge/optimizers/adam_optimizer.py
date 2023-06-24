import numpy as np

class AdamOptimizer:

    def __init__(self, layer_dims, learning_rate=1e-3, beta_m=0.99, beta_v=0.999, epsilon=1e-8):
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