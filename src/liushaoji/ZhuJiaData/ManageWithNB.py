from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import pandas
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
data = preprocessing.minmax_scale(data, feature_range=(0, 1)) #  normalization
# label
label = inputData[:, 3]

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.1, random_state=0)

names = ['GaussianNB', 'MultinomialNB', 'BernoulliNB']
classifiers = [GaussianNB(), MultinomialNB(), BernoulliNB()]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    print "%s = %f" % (name, score)
    score1 = cross_val_score(clf, data, label)
    print "%s = %f" % (name, np.mean(score1))

    # # plot learn curve
    # title = "Learning Curves (" + name + ")"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    # plc.plot_learning_curve_with_datasize(clf, title, data, label, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    # plt.show()


# test score
# corss val score
# GaussianNB = 0.362416
# GaussianNB = 0.433768
# MultinomialNB = 0.510067
# MultinomialNB = 0.474788
# BernoulliNB = 0.489933
# BernoulliNB = 0.464708