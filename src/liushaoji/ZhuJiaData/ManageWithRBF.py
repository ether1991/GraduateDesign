# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas

names = ["Nearest Neighbors", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier()]

inputFile = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/stl_block_r3.csv');
inputFile.dropna(inplace=True)  # drop the NaN records
inputData = inputFile.values
# data feature
data = inputData[:, 5:13]
data = preprocessing.scale(data) #  normalization
# label
label = inputData[:, 3]
seed=7

# iterate over classifiers
for name, clf in zip(names, classifiers):
    kfold = KFold(n=len(data), n_folds=10, shuffle=True, random_state=seed)
    results = cross_val_score(clf, data, label, cv=kfold)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
