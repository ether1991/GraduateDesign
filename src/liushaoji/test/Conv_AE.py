'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''
from __future__ import print_function
import numpy as np
import pandas
# np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Convolution1D, MaxPooling1D, Flatten


# Embedding
max_features = 1489
maxlen = 8
embedding_size = 128

# Convolution
filter_length = 3
nb_filter = 32
pool_length = 2

# Training
batch_size = 32
nb_epoch = 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

# dataframe = pandas.read_csv("/Users/liushaoji/PycharmProjects/GraduateDesign/file/day001.csv", header=0)
dataframe = pandas.read_csv("/Users/liushaoji/PycharmProjects/GraduateDesign/file/stl_block_r1.csv", header=None)
data = dataframe.values

# iris_data = pandas.read_csv("/Users/liushaoji/ManageSAEData/iris.txt", header=None )
# data = iris_data.values
X_train = data[:, 0:8]
# print(X_train)
y_train = data[:, 8]
# print(y_train)
print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
# model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='same',
                        activation='relu'))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(Dropout(0.2))

model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(256))
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train)

# score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)
