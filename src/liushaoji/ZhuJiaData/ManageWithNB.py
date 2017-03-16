from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import pandas
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import PlotLearnCurve as plc
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

inputFile = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/stl_block_r3.csv')
inputFile.dropna(inplace=True)  # drop the NaN records
inputData = inputFile.values
# data feature
data = inputData[:, 5:13]
# data = preprocessing.minmax_scale(data, feature_range=(0, 1)) #  normalization
# label
label = inputData[:, 3]
seed=7

names = ['GaussianNB', 'MultinomialNB', 'BernoulliNB']
classifiers = [GaussianNB(), MultinomialNB(), BernoulliNB()]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    kfold = KFold(n=len(data), n_folds=10, shuffle=True, random_state=seed)
    results = cross_val_score(clf, data, label, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
