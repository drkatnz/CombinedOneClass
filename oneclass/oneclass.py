# -*- coding: utf-8 -*-
"""
One-class classifer.

For more information see:
Kathryn Hempstalk, Eibe Frank, Ian H. Witten: One-Class Classification by Combining Density and Class Probability Estimation. In: Proceedings of the 12th European Conference on Principles and Practice of Knowledge Discovery in Databases and 19th European Conference on Machine Learning, ECMLPKDD2008, Berlin, 505--519, 2008.

@author: Kat
"""
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from generators import abstract,gaussian
from scipy import stats
import math


class OneClassClassifier(BaseEstimator):
    def __init__(self,base_classifier=DecisionTreeClassifier(),\
        contamination=0.1,proportion_generated=0.5,\
        cv_folds=10,\
        density_only=False,random_state=0):
            
        self.base_classifier = base_classifier
        self.contamination = contamination
        self.proportion_generated = proportion_generated
        self.density_only = density_only
        self.random_state = random_state
        self.cv_folds = cv_folds
        
        self.threshold = 0.5
    
    
    def fit(self,X,y=None):
        #create the data generators
        self.generators = [None] * X.shape[1]
        for col in xrange(X.shape[1]):
            mean = np.mean(X[:,col])
            stddev = np.std(X[:,col])           
            if(stddev == 0):
                generator = abstract.DummyGenerator(mean)
            else:
                generator = gaussian.GaussianGenerator(mean,stddev,self.random_state)
            self.generators[col] = generator
        
    
        #generate data        
        totalInstances = len(X) / (1 - self.proportion_generated)
        generated_len = int(totalInstances - len(X))
        generated = [None] * generated_len
        for i in xrange(generated_len):
            row = [None] * X.shape[1]
            for col in xrange(X.shape[1]):
                row[col] = self.generators[col].generate()
                generated[i] = row
             
        
        #work out the threshold of prob(X|C) using cross validation
        skf = StratifiedKFold(n_splits=self.cv_folds,\
            random_state=self.random_state, shuffle=True)
            
        newX = np.vstack((X,generated))
        newY = np.hstack((np.ones(len(X)),np.zeros(len(X))))
    
        thresholds = [None] * self.cv_folds
        for i, (train_indices, test_indices) in enumerate(skf.split(newX,newY)):
            if(~self.density_only):
                #only train if you need to!
                self.base_classifier.fit(newX[train_indices], newY[train_indices])
            
            probabilities = self._get_log_probabilities(newX[test_indices])                       
            thresholds[i] = stats.scoreatpercentile(probabilities, 100 * self.contamination)

        self.threshold = np.mean(thresholds)
                
        #retrain on all the data
        if(~self.density_only):
            self.base_classifier.fit(newX,newY)
        
       
    def _get_log_probabilities(self,X):
        probabilities = [None] * len(X)
        if(self.density_only):
            for i,x in enumerate(X):
                probabilities[i] = self._log_prob_x_given_a(x)              
        else:
            base_classifier_probs = self.base_classifier.predict_proba(X)[:,1]
            for i,x in enumerate(X):
                probabilities[i] = self._log_prob_x_given_c(x,base_classifier_probs[i])
            
        return np.array(probabilities)
        
        
    def _get_probabilities(self,X):
        log_probs = self._get_log_probabilities(X)            
        
        probabilities = [None] * len(X)
        for i,prob in enumerate(log_probs):
            if(prob == np.inf):
                prob_outlier = 0
            else:
                prob_outlier = 1 / (1 + math.exp(prob - self.threshold))
            prob_class = 1 - prob_outlier
            probabilities[i] = prob_class
            
        return np.array(probabilities)
        
        
    def _log_prob_x_given_a(self,x):
         prob_x_given_a = 0
         for col in xrange(x.shape[0]):             
             prob_x_given_a = prob_x_given_a + self.generators[col].get_log_probability(x[col])
           
         return prob_x_given_a
            
            
    def _log_prob_x_given_c(self,x,prob_c_given_x):
        prob_c = 1 - self.proportion_generated
        log_prob_x_given_a = self._log_prob_x_given_a(x)
        
        if(self.density_only):
            return log_prob_x_given_a
            
        #cover edge cases
        if(prob_c_given_x == 1):
            return np.inf
            
        if(prob_c_given_x == 0):
            return log_prob_x_given_a
            
        #finally, calculate probability
        top = math.log(1 - prob_c) + math.log(prob_c_given_x)
        bottom = math.log(prob_c) + math.log(1 - prob_c_given_x)
        
        return (top - bottom) + log_prob_x_given_a
        
    
    def predict(self,X):
        log_probs = self._get_log_probabilities(X)            
        
        probabilities = [None] * len(X)
        for i,prob in enumerate(log_probs):
            if prob >= self.threshold:
                probabilities[i] = 1
            else:
                probabilities[i] = -1
                
        return probabilities
    
    
    def decision_function(self,X):
        return self._get_probabilities(X)
    
