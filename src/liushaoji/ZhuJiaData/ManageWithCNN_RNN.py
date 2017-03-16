# CNN for the IMDB problem
import pandas
import numpy
from numpy import *
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.cross_validation import KFold
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout
from keras.constraints import maxnorm
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/stl_block_r3.csv')
dataframe.dropna(inplace=True)#drop the null records
dataset = dataframe.values

data = dataset[:, 5:13].astype(float)
label = dataset[:, 3]
print np.max(label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(label, 4)
seed=7
# X_train, X_test, Y_train, Y_test = train_test_split(data, dummy_y, test_size=0.2, random_state=0)
# define baseline model
def mymodel():
    inputData = Input(shape=(8,))

    encode = Dense(16, activation='tanh')(inputData)
    Dropout(0.1)
    encode = Dense(12, activation='tanh')(encode)
    Dropout(0.1)
    decode = Dense(8, activation='tanh')(encode)
    decode = Dense(4, activation='softmax')(encode)
    # decode = Dense(4, activation='sigmoid')(decode)
    autoencoder = Model(input=inputData, output=decode)
    autoencoder.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return autoencoder

estimator = KerasClassifier(build_fn=mymodel, nb_epoch=200, batch_size=5,verbose=1)
print len(dummy_y)
kfold = KFold(n=len(data), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, data, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# acc : 0.93