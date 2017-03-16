from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import pandas
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
import PlotLearnCurve as plc
inputFile = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/stl_block_r3.csv')
inputFile.dropna(inplace=True)  # drop the NaN records
inputData = inputFile.values
# data feature
data = inputData[:, 5:13]
# data = preprocessing.scale(data) #  normalization
# label
label = inputData[:, 3]
seed=7

clf = svm.SVC(kernel='rbf', C=1)  # class  It must be one of 'linear', 'poly', ('rbf', 'sigmoid'), 'precomputed'


kfold = KFold(n=len(data), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(clf, data, label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))