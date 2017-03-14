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
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/Day01_format_simply.csv')
dataframe.dropna(inplace=True)#drop the null records
dataset = dataframe.values

X = dataset[0:10000, 1:8]
Y = dataset[0:10000, 8]
Y = to_categorical(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
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
    model.add(Dense(4, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

mymodel = baseline_model()
mymodel.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
score = mymodel.evaluate(X_test, Y_test, batch_size=16)
