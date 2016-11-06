# -*- coding: utf-8 -*-
"""
Abstract interfaces/classes for one-class data generators.

@author: Kat
"""
from abc import abstractmethod
from random import Random 
import math

# Base class for all generators used for one-class classification.
class Generator:
    @abstractmethod
    def get_probability(self, value):
        pass
    @abstractmethod
    def get_log_probability(self, value):
        pass
    @abstractmethod
    def generate(self):
        pass

# Base class for generators that are randomizable.
class RandomizableGenerator:
    def __init__(self, seed=0):
        self.random_generator = Random()
        self.random_generator.seed(seed)
		
    def seed(self, value):
        self.random_generator.seed(value)
		
# Base class for generators that are ranged.
class RangedGenerator:
    def __init__(self,lower,upper):
        self.lower = lower
        self.upper = upper
		
    def set_lower(self, lower):
        self.lower = lower
	
    def set_upper(self, upper):
        self.upper = upper
		

# Base class for generators that utilise a mean and standard deviation.
class Mean:
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev
		
    def set_mean(self, mean):
        self.mean = mean
    		
    def set_stddev(self, stddev):
        self.stddev = stddev
	

class DummyGenerator(Generator):
    def __init__(self, value):
        self.value = value
        
    def get_probability(self, value):
        return 1

    def get_log_probability(self, value):
        return math.log(1)

    def generate(self):
        return self.value	
