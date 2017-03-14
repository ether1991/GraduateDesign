from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas

inputFile = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/Day01_format_simply.csv');
inputFile.dropna(inplace=True)  # drop the NaN records
inputData = inputFile.values
# data feature
data = inputData[0:2000, 1:8]
# label
label = inputData[0:2000, 8]

inputDim = 7

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=0)

# Y_train = to_categorical(Y_train)
# Y_test = to_categorical(Y_test)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Embedding(450000, 1, input_length=7, init='uniform'))
model.add(Dense(64, input_dim=inputDim, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          nb_epoch=20,
          batch_size=16)
score = model.evaluate(X_test, Y_test, batch_size=16)