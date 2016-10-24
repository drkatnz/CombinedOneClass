# -*- coding: utf-8 -*-
"""
Gaussian generators for one-class data generation.

@author: Kat
"""
import math
import scipy.stats as stats
from abstract import RandomizableGenerator,Mean

class GaussianGenerator(RandomizableGenerator,Mean):
    def __init__(self,mean,stddev,seed=0):
        Mean.__init__(self,mean,stddev)
        RandomizableGenerator.__init__(self,seed)
        
    def get_probability(self, value):
        return stats.norm(self.mean,self.stddev).pdf(value)
    
    def get_log_probability(self, value):
        return math.log(self.get_probability(value))
    
    def generate(self):
        return self.random_generator.gauss(self.mean,self.stddev)
