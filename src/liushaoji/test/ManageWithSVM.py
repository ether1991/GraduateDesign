from sklearn import svm
import pandas
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
import numpy as np
inputFile = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/Day01_format.csv');
inputFile.dropna(inplace=True)  # drop the NaN records
inputData = inputFile.values
# data feature
data = inputData[:, 1:8]
# label
label = inputData[:, 8]

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.3, random_state=0)

clf = svm.SVC(kernel='linear', C=1) # class  It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
clf.fit(X_train, Y_train)  # training the svc model
#
# result = clf.predict([7.7,2.8,6.7,2.0])  # predict the target of testing samples
# print result  # target
# acc = clf.score(X, Y)
# print acc

scores = cross_val_score(clf, X_test, Y_test)
print scores
#
# print clf.support_vectors_  # support vectors

# print clf.support_  # indeices of support vectors

# print clf.n_support_  # number of support vectors for each class