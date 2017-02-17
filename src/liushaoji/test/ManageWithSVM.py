from sklearn import svm
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
inputFile = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/Day01_format_simply.csv');
inputFile.dropna(inplace=True)  # drop the NaN records
inputData = inputFile.values
# data feature
data = inputData[:, 1:8]
# label
label = inputData[:, 8]

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.1, random_state=0)

clf = svm.SVC(kernel='rbf', C=1)  # class  It must be one of 'linear', 'poly', ('rbf', 'sigmoid'), 'precomputed'
clf.fit(X_train, Y_train)  # training the svc model

scores = cross_val_score(clf, X_test, Y_test)

print scores
print np.mean(scores)
print np.std(scores)