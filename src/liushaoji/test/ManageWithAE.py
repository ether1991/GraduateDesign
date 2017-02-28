import pandas
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.utils.visualize_util import plot

# load dataset
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/Day01_format_simply.csv')
dataframe.dropna(inplace=True)#drop the null records
dataset = dataframe.values

data = dataset[0:100, 1:8]
label = dataset[0:100, 8]

encoder = LabelEncoder()
encoder.fit(label)
encoded_Y = encoder.transform(label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, Y_train, Y_test = train_test_split(data, dummy_y, test_size=0.3, random_state=0)
# define baseline model

inputData = Input(shape=(7,))

encode = Dense(16, activation='tanh')(inputData)
encode = Dense(12, activation='tanh')(encode)
decode = Dense(8, activation='sigmoid')(encode)
decode = Dense(1, activation='sigmoid')(encode)
# decode = Dense(4, activation='sigmoid')(decode)

autoencoder = Model(input=inputData, output=decode)

autoencoder.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['recall'])

autoencoder.fit(X_train, Y_train, batch_size=16, nb_epoch=20, verbose=2)

score, acc = autoencoder.evaluate(X_test, Y_test, verbose=2)
# print score
# print acc
# plot(autoencoder, to_file='model.png')