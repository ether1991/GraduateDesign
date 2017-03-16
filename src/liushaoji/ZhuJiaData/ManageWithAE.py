import pandas
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.utils.visualize_util import plot

# load dataset
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/stl_block_r3.csv')
dataframe.dropna(inplace=True)#drop the null records
dataset = dataframe.values

data = dataset[:, 5:13].astype(float)
label = dataset[:, 3]

seed = 7
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(label)

def mymodel():
    model = Sequential()
    model.add(Dense(16, input_dim=8))
    model.add(Dropout(0.1))
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(4, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



estimator = KerasClassifier(build_fn=mymodel, nb_epoch=200, batch_size=5,verbose=1)
print len(dummy_y)
kfold = KFold(n=len(data), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, data, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))