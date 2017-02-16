import pandas
from keras.layers import Dense, Input, Dropout
from keras.models import Model

iris_data = pandas.read_csv("/Users/liushaoji/PycharmProjects/GraduateDesign/file/stl_block_r1.csv", header=None )
data = iris_data.values

iris = data[0:, :8]
iris_test = data[23:100, :8]
# print iris
label = data[:, 8]
label_test = data[23:100, 8]
# print label

inputData = Input(shape=(8,))

encode = Dense(10, activation='relu')(inputData)
encode = Dense(6, activation='relu')(encode)
encode = Dense(3, activation='relu')(encode)
decode = Dense(1, activation='softmax')(encode)
# decode = Dense(4, activation='sigmoid')(decode)

autoencoder = Model(input=inputData, output=decode)

autoencoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
autoencoder.fit(iris, label, batch_size=16, nb_epoch=20, verbose=2)

# score, acc = autoencoder.evaluate(iris_test, label_test, verbose=2)
# print score
# print acc