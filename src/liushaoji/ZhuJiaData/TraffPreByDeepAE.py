import pandas
from keras.layers import Dense, Input
from keras.models import Model

dataset = pandas.read_csv("/Users/liushaoji/PycharmProjects/GraduateDesign/file/day01.csv", header=0)
data = dataset.values
line0_Max = max(data[0:, 0])
line1_Max = max(data[0:, 1])
line2_Max = max(data[0:, 2])
line3_Max = max(data[0:, 3])
line4_Max = max(data[0:, 4])
data[0:, 0] = data[0:, 0]/line0_Max
data[0:, 1] = data[0:, 1]/line1_Max
data[0:, 2] = data[0:, 2]/line2_Max
data[0:, 3] = data[0:, 3]/line3_Max
data[0:, 4] = data[0:, 4]/line4_Max
data[0:, 5] = data[0:, 5]*10

trainData = data[0:, :5]
# print trainData
trainLabel = data[:, 5]
# print trainLabel

inputData = Input(shape=(5,))

encode = Dense(5, activation='tanh')(inputData)
encode = Dense(3, activation='tanh')(encode)
decode = Dense(1, activation='sigmoid')(encode)

autoencoder = Model(input=inputData, output=decode)

autoencoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
autoencoder.fit(trainData, trainLabel, batch_size=32, nb_epoch=100, verbose=2)