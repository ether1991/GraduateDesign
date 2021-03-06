# CNN for the IMDB problem
import pandas
import numpy
from numpy import *
from keras.models import Sequential
from keras.layers import Dense
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
from keras.utils.visualize_util import plot

seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/Day01_format.csv')
dataframe.dropna(inplace=True)#drop the null records
dataset = dataframe.values

X = dataset[:, 1:8].astype(float)
Y = dataset[:, 8]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y, 10)
print dummy_y.shape
# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    encoding_dim = 7

    encoded = Dense(encoding_dim, activation='sigmoid')

    model.add(Embedding(450000, 1, input_length=7, init='uniform'))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(encoded)
    model.add(Dense(256, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dense(10, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # plot(model, to_file='model.png')
    return model
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=10, batch_size=16, verbose=2)
print len(dummy_y)
kfold = KFold(n=len(X), n_folds=4, shuffle=True, random_state=seed)#n_folds=10
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))