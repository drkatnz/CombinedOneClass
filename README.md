# CombinedOneClass
One-class classifier that combines density and class probability estimation. 
This is a Python implementation of the OneClassClassifier library from WEKA.

For more information see:
Kathryn Hempstalk, Eibe Frank, Ian H. Witten: [One-Class Classification by Combining Density and Class Probability Estimation](https://github.com/drkatnz/CombinedOneClass/blob/master/hfw08-oneclassclassification.pdf). In: Proceedings of the 12th European Conference on Principles and Practice of Knowledge Discovery in Databases and 19th European Conference on Machine Learning, ECMLPKDD2008, Berlin, 505--519, 2008.

Currently still a **work-in-progress** and the code is untested.

## Pre-requisites
```
scikit-learn >= 18.0
numpy >= 1.11
scipy >= 0.17.1
```
Written for python 2.7 support, not checked with python 3.

## Usage example
```
import sys
sys.path.append('path/to/repository/locally/oneclass')
import oneclass

occ = oneclass.OneClassClassifier(contamination=outliers_fraction)
occ.fit(X)

# will predict 1 for inlier, -1 for outlier
occ.predict(X) 

# will return an array, each item is the probability of being an inlier.
occ.decision_function(X)
```

### Supported functionality

At this stage just the Gaussian generator is supported for the data generation step. This is the default in WEKA.
If the standard deviation is zero (all values the same) a dummy generator will be used instead of a gaussian.

### TODO

- Weights are not supported.
- Base classifier has been set to scikit-learn's DecisionTreeClassifier, this should be better optimised.
- Uniform data generators not supported.