from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import pandas
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
data = preprocessing.scale(data) #  normalization
# label
label = inputData[:, 3]

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.1, random_state=0)

clf = svm.SVC(kernel='rbf', C=1)  # class  It must be one of 'linear', 'poly', ('rbf', 'sigmoid'), 'precomputed'
clf.fit(X_train, Y_train)  # training the svc model
# print clf.predict([43200,400060,37.870977,122.305289,181.0,75,64.0])

scores = cross_val_score(clf, data, label)  # cross validation
#
print scores
print np.mean(scores)
print np.std(scores)

# plot learn curve
title = "Learning Curves (SVM)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# plc.plot_learning_curve_with_datasize(clf, title, data, label, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
# plt.show()

# result
# [ 0.6733871   0.67741935  0.74141414]
# 0.697406864342
# 0.0311613553699