import pandas
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.constraints import maxnorm

# load dataset
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/stl_block_r3.csv')
dataframe.dropna(inplace=True)#drop the null records
dataset = dataframe.values

data = dataset[:, 5:13]
label = dataset[:, 3]

encoder = LabelEncoder()
encoder.fit(label)
encoded_Y = encoder.transform(label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, Y_train, Y_test = train_test_split(data, dummy_y, test_size=0.3, random_state=0)
# define baseline model

inputData = Input(shape=(8,))

model = Sequential()
encoding_dim = 7

encoded = Dense(encoding_dim, activation='sigmoid')

model.add(Embedding(450000, 1, input_length=8, init='uniform'))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', W_constraint=maxnorm(3)))
model.add(Dense(3, init='normal', activation='softmax'))

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['recall'])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, nb_epoch=20, verbose=2)

score, acc = model.evaluate(X_test, Y_test, verbose=2)
# print score
# print acc
# plot(autoencoder, to_file='model.png')