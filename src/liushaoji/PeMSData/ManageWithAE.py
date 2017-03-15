import pandas
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.cross_validation import KFold
from keras.optimizers import RMSprop
# from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.visualize_util import plot

# load dataset
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/Day01_format.csv')
dataframe.dropna(inplace=True)#drop the null records
dataset = dataframe.values

data = dataset[0:10000, 1:8]
label = dataset[0:10000, 8]

encoder = LabelEncoder()
encoder.fit(label)
encoded_Y = encoder.transform(label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y, 5)
print dummy_y.shape
seed=7
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.1, random_state=0)
Y_train = np_utils.to_categorical(Y_train, 5)
# Y_test = np_utils.to_categorical(Y_test, 5)
# define baseline model
def mymodel():
    inputData = Input(shape=(7,))

    encode = Dense(32, activation='tanh')(inputData)
    Dropout(0.1)
    encode = Dense(16, activation='tanh')(encode)
    # Dropout(0.1)
    encode = Dense(8, activation='tanh')(encode)
    decode = Dense(2, activation='sigmoid')(encode)
    # decode = Dense(2, activation='sigmoid')(encode)
    # decode = Dense(4, activation='sigmoid')(decode)

    autoencoder = Model(input=inputData, output=decode)

    optimizer = RMSprop(lr=0.002, rho=0.9, epsilon=1e-06)
    # autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    autoencoder.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

    autoencoder.fit(X_train, Y_train, batch_size=8, nb_epoch=20, verbose=1)

    # score, acc = autoencoder.evaluate(X_test, Y_test, verbose=2)
    # print score
    # print acc
    # plot(autoencoder, to_file='model.png')
    return autoencoder

# mymodel = mymodel()
# mymodel.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
# score = mymodel.evaluate(X_test, Y_test, batch_size=16)

estimator = KerasClassifier(build_fn=mymodel, nb_epoch=200, batch_size=5, verbose=2)
print len(dummy_y)
kfold = KFold(n=len(data), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, data, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))