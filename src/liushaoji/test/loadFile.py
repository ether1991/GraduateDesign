import pandas
from keras.layers import Dense, Input, Convolution1D, MaxPooling1D, UpSampling1D
from keras.models import Model

# dataframe = pandas.read_csv("/Users/liushaoji/ManageSAEData/day1.csv", header=0)
# dataset = dataframe.values
#
# data = dataset[0:, 0:8]
# label = dataset[0:, 8]

# iris_data = pandas.read_csv("/Users/liushaoji/ManageSAEData/iris.txt", header=None )
# data = iris_data.values

# iris = data[0:, :4]
# print iris
# label = data[:, 4]
# print label


inputData = Input(shape=(5,))

encode = Convolution1D(3, activation='tanh')(inputData)
encode = MaxPooling1D(2)(encode)
encode = Convolution1D(3, activation='tanh')(encode)

encoded = MaxPooling1D(2)(encode)

decode = Convolution1D(3, activation='tanh')(encoded)
decode = UpSampling1D(2)(decode)
decode = Convolution1D(3, activation='tanh')(decode)
decode = UpSampling1D(2)(decode)

decoded = Convolution1D(5, activation='sigmoid', border_mode='same')(decode)

autoencoder = Model(input=inputData, output=decoded)

autoencoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

autoencoder.fit(dataset, label, batch_size=32, nb_epoch=200, verbose=2)

# score, acc = autoencoder.evaluate(iris, label, verbose=0)
#
# print score
# print acc