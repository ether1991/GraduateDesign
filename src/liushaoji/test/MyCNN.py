import pandas
from keras.layers import Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding

# set parameters:
max_features = 500000
maxlen = 5
batch_size = 32
embedding_dims = 300
# nb_filter = 10
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 10

dataframe = pandas.read_csv("/Users/liushaoji/ManageSAEData/day01.csv", header=0)
data = dataframe.values
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

# iris_data = pandas.read_csv("/Users/liushaoji/ManageSAEData/iris.txt", header=None )
# data = iris_data.values

X_train = data[0:, :5]
y_train = data[:, 5]

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen,
                    dropout=0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='tanh',
                        subsample_length=1))
# we use max pooling:
model.add(MaxPooling1D(pool_length=model.output_shape[1]))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('tanh'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,  batch_size=batch_size,  nb_epoch=nb_epoch)