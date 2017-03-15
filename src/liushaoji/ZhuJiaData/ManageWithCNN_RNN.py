# CNN for the IMDB problem
import pandas
import numpy
from numpy import *
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
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/Day01_format_simply.csv')
dataframe.dropna(inplace=True) #drop the null records
dataset = dataframe.values

data = dataset[:, 5:13]
label = dataset[:, 3]

encoder = LabelEncoder()
encoder.fit(label)
encoded_Y = encoder.transform(label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
seed=7
X_train, X_test, Y_train, Y_test = train_test_split(data, dummy_y, test_size=0.2, random_state=0)
# define baseline model
def mymodel():
    inputData = Input(shape=(7,))

    encode = Dense(16, activation='tanh')(inputData)
    Dropout(0.1)
    encode = Dense(12, activation='tanh')(encode)
    Dropout(0.1)
    decode = Dense(8, activation='tanh')(encode)
    decode = Dense(4, activation='sigmoid')(encode)
    # decode = Dense(4, activation='sigmoid')(decode)

    autoencoder = Model(input=inputData, output=decode)

    autoencoder.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    autoencoder.fit(X_train, Y_train, batch_size=8, nb_epoch=20, verbose=1)

    # score, acc = autoencoder.evaluate(X_test, Y_test, verbose=2)
    # print score
    # print acc
    # plot(autoencoder, to_file='model.png')
    return autoencoder

estimator = KerasClassifier(build_fn=mymodel, nb_epoch=200, batch_size=5,verbose=2)
print len(dummy_y)
kfold = KFold(n=len(data), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, data, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# acc : 0.93