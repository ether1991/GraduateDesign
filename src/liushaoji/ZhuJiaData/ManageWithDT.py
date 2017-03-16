from sklearn import tree
import pandas
import numpy as np
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import KFold
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
# data = preprocessing.scale(data) #  normalization
# label
label = inputData[:, 3]
seed=7
# label = np_utils.to_categorical(label)

clf = tree.DecisionTreeClassifier()

kfold = KFold(n=len(data), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(clf, data, label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))