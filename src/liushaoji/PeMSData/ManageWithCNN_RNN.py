import pandas
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout
from keras.constraints import maxnorm
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import KFold
# from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/Day01_format.csv')
dataframe.dropna(inplace=True)
dataset = dataframe.values

X = dataset[0:10000, 1:8]
Y = dataset[0:10000, 8]

np_utils.to_categorical(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(Y, 5)
seed=7

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# define baseline model
def mymodel():
    # create model
    model = Sequential()
    encoding_dim = 7

    encoded = Dense(encoding_dim, activation='sigmoid')
    model.add(Embedding(450000, 1, input_length=7, init='uniform'))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(encoded)
    model.add(Dense(256, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dense(5, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

estimator = KerasClassifier(build_fn=mymodel, nb_epoch=200, batch_size=5,verbose=2)
print len(dummy_y)
kfold = KFold(n=len(X), n_folds=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))