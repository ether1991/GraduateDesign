# CNN for the IMDB problem
import pandas
# import random
import numpy
# from keras.optimizers import SGD
from numpy import * 
# from keras.models import Model
# from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Input, Dense
# from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D, UpSampling1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence
from keras.utils import np_utils
# from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.cross_validation import KFold
# from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout
# from ruamel_yaml.main import enc
from keras.constraints import maxnorm

seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/dayFormat.csv')
dataframe.dropna(inplace=True)
dataset = dataframe.values
X = dataset[:, 1:8].astype(float)
Y = dataset[:, 8]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# define baseline model
def baseline_model(): 
# create model
    model = Sequential()
    encoding_dim = 7
   
    encoded = Dense(encoding_dim,activation='sigmoid') 
    #encoded = Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu')

    
   # model.add(Dense(8, input_dim=8, init='normal', activation='relu'))
    model.add(Embedding(101, 1, input_length=7, init='uniform'))
   # model.add(Dropout(0.2))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
 
    model.add(MaxPooling1D(pool_length=2))
   
   
    
    #model.add(Convolution1D(nb_filter=16, filter_length=3, border_mode='same', activation='relu'))
 
    #model.add(MaxPooling1D(pool_length=2))
   
    model.add(Dropout(0.2))
    
        
    #model.add(Convolution1D(nb_filter=8, filter_length=3, border_mode='same', activation='relu'))
 
    #model.add(MaxPooling1D(pool_length=2))
    
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    
   # model.add(Flatten())
    #model.add(Dense(8, input_dim=8, init='normal', activation='relu'))
    model.add(encoded) 
   
    #model.add(MaxPooling1D())
    #model.add(LSTM(100,return_sequences=False))
    #model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu',W_constraint=maxnorm(3)))
   # model.add(Dense(256, activation='sigmoid',W_constraint=maxnorm(3)))
    #model.add(Dropout(0.2))
    #model.add(Dense(4, init='normal', activation='relu'))
  #  model.add(Embedding(8, 32, input_length=8))
   # model.add(LSTM(100))
   
    model.add(Dense(8, init='normal', activation='softmax'))
# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
   # model = Sequential()
   # model.add(Embedding(8, 32, input_length=8))
   # model.add(Dropout(0.2))
    #model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
 
    #model.add(MaxPooling1D(pool_length=2))
    #model.add(Dropout(0.2))
    #model.add(LSTM(100))
    #model.add(Dropout(0.2))
    #model.add(Flatten())
    #model.add(Dense(8, activation='sigmoid'))
    #model.add(Dropout(0.5))
    #model.add(Dense(3, activation='softmax'))
    #epochs = 100
    #lrate = 0.01
    #decay = lrate/epochs
   # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=2)
print len(dummy_y)
kfold = KFold(n=len(X), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))