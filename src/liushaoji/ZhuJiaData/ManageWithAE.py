import pandas
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.utils.visualize_util import plot

# load dataset
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/stl_block_r3.csv')
dataframe.dropna(inplace=True)#drop the null records
dataset = dataframe.values

data = dataset[:, 5:13].astype(float)
label = dataset[:, 3]

seed = 7
encoder = LabelEncoder()
encoder.fit(label)
encoded_Y = encoder.transform(label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_test, Y_train, Y_test = train_test_split(data, dummy_y, test_size=0.2, random_state=0)
# define baseline model
def mymodel():
    model = Sequential()

    model.add(Dense(16, input_dim=8))
    model.add(Dropout(0.1))
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(4, activation='sigmoid'))
    # decode = Dense(4, activation='sigmoid')(decode)

    # autoencoder = Model(input=inputData, output=decode)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    # autoencoder.fit(X_train, Y_train, batch_size=16, nb_epoch=20, verbose=2)
    #
    # score, acc = autoencoder.evaluate(X_test, Y_test, verbose=2)
    # print score
    # print acc
    # plot(autoencoder, to_file='model.png')

    # result  ========  acc : 0.82


estimator = KerasClassifier(build_fn=mymodel, nb_epoch=200, batch_size=5,verbose=2)
print len(dummy_y)
kfold = KFold(n=len(data), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, data, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))