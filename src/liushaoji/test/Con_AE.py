from keras.layers import Input, Dense
from keras.models import Model
import pandas

dataframe = pandas.read_csv("/Users/liushaoji/ManageSAEData/data.csv", header=0)
labelframe = pandas.read_csv("/Users/liushaoji/ManageSAEData/label.csv", header=0)
data = dataframe.values
label = labelframe.values

input_img = Input(shape=(5,))
encoded = Dense(10, activation='relu')(input_img)
encoded = Dense(5, activation='relu')(encoded)
encoded = Dense(3, activation='relu')(encoded)

decoded = Dense(5, activation='relu')(encoded)
decoded = Dense(10, activation='relu')(decoded)
decoded = Dense(5, activation='sigmoid')(decoded)

autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(data, data)