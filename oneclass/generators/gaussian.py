# -*- coding: utf-8 -*-
"""
Gaussian generators for one-class data generation.

@author: Kat
"""
import math
import numpy as np
from abstract import RandomizableGenerator,Mean

class GaussianGenerator(RandomizableGenerator,Mean):
    def __init__(self,mean,stddev,seed=0):
        Mean.__init__(self,mean,stddev)
        RandomizableGenerator.__init__(self,seed)
        
    def get_probability(self, value):
        #doing math here for speed increase
        twopisqrt = math.sqrt(2 * math.pi)
        left = 1 / (self.stddev * twopisqrt)
        diffsquared = math.pow((value - self.mean), 2)
        bottomright = 2 * math.pow(self.stddev, 2)
        brackets = -1 * (diffsquared / bottomright)

        probx = left * math.exp(brackets)

        return probx

    
    def get_log_probability(self, value):
        probability = self.get_probability(value)
        if probability == 0:
            return np.inf
            
        return math.log(probability)
    
    def generate(self):
        return self.random_generator.gauss(self.mean,self.stddev)
