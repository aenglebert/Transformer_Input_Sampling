# Torch Implementation of Sparseness
# Inspired by the author initial implementation: 
# <https://github.com/jfc43/advex/blob/master/DNN-Experiments/Fashion-MNIST/utils.py>.
# from the paper "Concise Explanations of Neural Networks using Adversarial Training":
# <http://proceedings.mlr.press/v119/chalasani20a/chalasani20a.pdf>

import torch 

class Sparseness:
    def __init__(self, shift=False):
        """
        Init of Sparseness metric class
        :param shift: optional parameter to shift the minimum value of the 
        saliency map to zero.
        """
        self.shift = shift
        
    def __call__(self, saliency_map):
        
        if self.shift==True:
            # Set the minimum value of the saliency map to 0 
            saliency_map = saliency_map - saliency_map.amin()
            
        # Only positive non-zero values should be used.
        assert(saliency_map>=0).all() == True, """
        Non-positive values are not accepted for the Sparseness Metric (Gini Coefficient).
        One solution can be to shift the minimum value to zero 
        using self.shift=True
        """
    
        # Reshape saliency map to 1D-Vector
        flat_saliency = torch.flatten(saliency_map)
        
        # Avoid 0 values in the saliency map
        flat_saliency = torch.add(flat_saliency, 0.0000001)
        print("flat:", flat_saliency)
        # Length of the flattened tensor
        length = flat_saliency.size()[0]
        print("length:", length)
        
        
        # Sort the tensor in ascending order
        sorted_saliency = torch.sort(flat_saliency)
        print("sorted:", sorted_saliency)
        
        # Rank of each item of the sorted tensor: from 1 to length+1
        sorted_rank = torch.arange(start=1, end=length+1, step=1)
        print("rank:", sorted_rank)
        # Compute Sparseness 
        # (possible values :[0,1] with 0 perfect equality and 1 perfect inequality for an infini)
        # based on the simplification of the equation
        # of the Gini Coefficient of Inequality:
        # <https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm>.
        sparseness = torch.sum((sorted_rank * 2 - length - 1) * sorted_saliency.values) / (length * torch.sum(sorted_saliency.values))  
        
        return sparseness