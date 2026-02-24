import numpy as np

class Loss():
    def __init__(self):
        pass
    
    def softmax_crossentropy(self, logits, label_true=1, eps=1e-12):
        # Softmax
        exp_vals = np.exp(logits - np.max(logits))
        self.soft_max = exp_vals / np.sum(exp_vals)
        softmax_array = np.clip(self.soft_max, eps, 1.0) # to avoid log(0) in cross-entropy calculation

        # Create probability vector for true label
        self.label_prob = np.zeros_like(self.soft_max)
        self.label_prob[label_true, 0] = 1

        # Cross entropy
        self.loss = -np.sum(self.label_prob * np.log(softmax_array))

        return self.loss
    
    def backward(self):
        return np.asarray(self.soft_max-self.label_prob)
    


    
