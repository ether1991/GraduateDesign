from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import pandas
from sklearn.model_selection import train_test_split
import numpy as np

inputFile = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/stl_block_r3.csv')
inputFile.dropna(inplace=True)  # drop the NaN records
inputData = inputFile.values
# data feature
data = inputData[:, 5:13]
# label
label = inputData[:, 3]

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.1, random_state=0)

data_dim = 7

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(Embedding(450000, 1, input_length=8, init='uniform'))
model.add(LSTM(128, activation='sigmoid', inner_activation='hard_sigmoid')) # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(16))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=64, nb_epoch=5,
          validation_data=(X_test, Y_test))