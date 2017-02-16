from sklearn import svm
import pandas
from sklearn.cross_validation import cross_val_score
import numpy as np
inputFile = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/dayFormat.csv');
inputFile.dropna(inplace=True)  # drop the NaN records
inputData = inputFile.values
# data feature
X = inputData[:, 1:8]
# label
Y = inputData[:, 8]

clf = svm.SVC(kernel='sigmoid')  # class  It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
clf.fit(X, Y)  # training the svc model
#
# result = clf.predict([7.7,2.8,6.7,2.0])  # predict the target of testing samples
# print result  # target
# acc = clf.score(X, Y)
# print acc

scores = cross_val_score(clf, X, Y)
print scores
#
# print clf.support_vectors_  # support vectors

# print clf.support_  # indeices of support vectors

# print clf.n_support_  # number of support vectors for each class