import numpy as np

class Loss():
    def __init__(self):
        pass
    
    def softmax(self, logits):
        exp_vals = np.exp(logits - np.max(logits))
        return exp_vals / np.sum(exp_vals)


loss = Loss()



