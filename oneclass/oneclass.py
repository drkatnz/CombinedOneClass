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
from generators import abstract, gaussian
from scipy import stats
import math

class OneClassClassifier(BaseEstimator):
    def __init__(self,base_classifier=DecisionTreeClassifier(),\
        outlier_fraction=0.1,proportion_generated=0.5,\
        cv_folds=10,\
        density_only=False,random_state=0):
            
        self.base_classifier = base_classifier
        self.outlier_fraction = outlier_fraction
        self.proportion_generated = proportion_generated
        self.density_only = density_only
        self.random_state = random_state
        self.cv_folds = cv_folds
        
        self.threshold = 0.5
    
    
    def fit(self,X,y=None):
        #create the data generators
        self.generators = [None] * X.shape[1]
        for col in xrange(X.shape[1]):
            generator = gaussian.GaussianGenerator(np.mean(X[:,col]),np.std(X[:,col]))
            self.generators[col] = generator
            
    
        #generate data
        generated = [None] * len(X)
        for i in xrange(len(X)):
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
            self.base_classifier.fit(newX[train_indices], newY[train_indices])
            probabilities = self._get_log_probabilities(newX[test_indices])
            
            thresholds[i] = stats.scoreatpercentile(probabilities, 100 * self.outlier_fraction)
            
        
        self.threshold = np.mean(thresholds)
        
        #retrain on all the data
        self.base_classifier.fit(newX,newY)
        
        
        
    def _get_log_probabilities(self,X):
        base_classifier_probs = self.base_classifier.predict_proba(X)[:,1]
        probabilities = [None] * len(X)
        for i,x in enumerate(X):
            probabilities[i] = self._prob_x_given_c(x,base_classifier_probs[i])
            
        return np.array(probabilities)
        
    def _get_probabilities(self,X):
        base_classifier_probs = self.base_classifier.predict_proba(X)[:,1]
        probabilities = [None] * len(X)
        for i,x in enumerate(X):
            prob = self._prob_x_given_c(x,base_classifier_probs[i])
            if(prob == np.inf):
                prob_outlier = 0
            else:
                prob_outlier = 1 / (1 + math.exp(prob - self.threshold))
            prob_class = 1 - prob_outlier
            probabilities[i] = prob_class
            
        return np.array(probabilities)
            
    def _prob_x_given_c(self,x,prob_c_given_x):
        prob_c = 1 - self.proportion_generated
        prob_x_given_a = 0
        for col in xrange(x.shape[0]):
            prob_x_given_a = prob_x_given_a + self.generators[col].get_log_probability(x[col])
        
        if(self.density_only):
            return prob_x_given_a
            
        if(prob_c_given_x == 1):
            return np.inf
            
        if(prob_c_given_x == 0):
            return prob_x_given_a
            
        #print prob_c, prob_c_given_x, math.exp(prob_x_given_a)
        top = math.log(1 - prob_c) + math.log(prob_c_given_x)
        bottom = math.log(prob_c) + math.log(1 - prob_c_given_x)
        
        return (top - bottom) + prob_x_given_a
        
    
        
    
    def predict(self,X):
        base_classifier_probs = self.base_classifier.predict_proba(X)[:,1]
        probabilities = [None] * len(X)
        for i,x in enumerate(X):
            prob = self._prob_x_given_c(x,base_classifier_probs[i])
            if prob >= self.threshold:
                probabilities[i] = 1
            else:
                probabilities[i] = -1
                
        return probabilities
    
    def decision_function(self,X):
        return self._get_probabilities(X)
    
