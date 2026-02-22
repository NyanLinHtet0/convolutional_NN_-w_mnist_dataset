import numpy as np

class Loss():
    def __init__(self):
        pass
    
    def softmax_crossentropy(self, logits, label_prob=[1,0,0], eps=1e-12):
        exp_vals = np.asarray(np.exp(logits - np.max(logits)))
        self.soft_max = exp_vals / np.sum(exp_vals)
        
        softmax_array = np.asarray(self.soft_max)
        self.label_prob = np.asarray(label_prob).reshape(softmax_array.shape)
        softmax_array = np.clip(softmax_array, eps, 1.0)  # to avoid log(0)
        loss = -np.sum(self.label_prob * np.log(softmax_array))
        return loss
    
    def backward(self):
        return np.asarray(self.soft_max-self.label_prob)
