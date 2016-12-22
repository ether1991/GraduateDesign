import pandas
from keras.layers import Dense, Input
from keras.models import Model

dataset = pandas.read_csv("/Users/liushaoji/ManageSAEData/PeMS_Data_WithWeek1.csv", header=0)
data = dataset.values

trainData = data[0:, :6]
print trainData
trainLabel = data[:, 6]
print trainLabel
inputData = Input(shape=(6,))

encode = Dense(5, activation='tanh')(inputData)
encode = Dense(3, activation='tanh')(encode)
decode = Dense(1, activation='sigmoid')(encode)

autoencoder = Model(input=inputData, output=decode)

autoencoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
autoencoder.fit(trainData, trainLabel, batch_size=32, nb_epoch=100, verbose=2)