import pandas
import numpy as np
from keras.layers import Dense, Input, Dropout
from sklearn.cross_validation import KFold
# from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm

# load dataset
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/Day01_format.csv')
dataframe.dropna(inplace=True)#drop the null records
dataset = dataframe.values

data = dataset[:, 1:8]
label = dataset[:, 8]
seed=7

label = np_utils.to_categorical(label)

def mymodel():
    inputData = Input(shape=(7,))

    model = Sequential()
    encoding_dim = 7

    encoded = Dense(encoding_dim, activation='sigmoid')

    model.add(Embedding(450000, 1, input_length=7, init='uniform'))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='tanh'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

estimator = KerasClassifier(build_fn=mymodel, nb_epoch=200, batch_size=5,verbose=2)
print len(label)
kfold = KFold(n=len(data), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, data, label, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))