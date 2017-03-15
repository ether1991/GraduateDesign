from sklearn import tree
import pandas
import numpy as np
import PlotLearnCurve as plc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

inputFile = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/stl_block_r3.csv')
inputFile.dropna(inplace=True)  # drop the NaN records
inputData = inputFile.values
# data feature
data = inputData[:, 5:13]
data = preprocessing.scale(data) #  normalization
# label
label = inputData[:, 3]

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=0)

clf = tree.DecisionTreeClassifier()
# 1
clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)
print score

# 2
scores = cross_val_score(clf, data, label)
print np.mean(scores)

# 3 plot learn curve
# title = "Learning Curves (DT)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# plc.plot_learning_curve_with_datasize(clf, title, data, label, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
# plt.show()

# test score : 0.835570469799
# cross val socre : 0.823145432823