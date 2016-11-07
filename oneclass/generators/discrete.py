# -*- coding: utf-8 -*-
"""
Discrete generators for one-class data generation.

@author: Kat
"""
import math
import numpy as np
from abstract import Generator, RandomizableGenerator

class DiscreteGenerator(Generator, RandomizableGenerator):
    def __init__(self,values,seed=0):
        RandomizableGenerator.__init__(self,seed)
        
        if(len(values) == 0):
            raise Exception('You must have at least one value for the dictionary!')
        
        #go through all the values and add them to a dictionary
        self._values_dict = {}
        for x in values:
            if x in self._values_dict:
                self._values_dict[x] = self._values_dict[x] + 1.0
            else:
                self._values_dict[x] = 1.0
                
        self._unseen_probability = 1.0 / float(len(values)) + 1.0
        self._total_values = float(len(values))
        self.total_keys = len(self._values_dict.keys())
        
        
    def get_probability(self, value):
        if value in self._values_dict:
            return self._values_dict[value] / (self._total_values + 1.0)

        return self._unseen_probability

    
    def get_log_probability(self, value):
        probability = self.get_probability(value)
        if probability == 0:
            return np.inf;
            
        return math.log(probability)
    
    
    def generate(self):
        num = self.random_generator.randint(0,self._total_values - 1)
        position = 0
        for key in self._values_dict.keys():
            thisval = self._values_dict[key]
            if(position + thisval >= num):
                return key
            else:
                position = position + thisval
        
        return self._values_dict.keys()[0]
