from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
from keras.layers import LSTM
import pandas
# from sklearn.model_selection import cross_val_score
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# import  PlotLearnCurve as plc
# from sklearn.model_selection import ShuffleSplit
# from keras.wrappers.scikit_learn import KerasClassifier

inputFile = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/Day01_format_simply.csv')
inputFile.dropna(inplace=True)  # drop the NaN records
inputData = inputFile.values
# data feature
data = inputData[0:10000, 1:8]
# label
label = inputData[0:10000, 8]
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.2, random_state=0)

max_features = 5
maxlen = len(X_train[0])

model = Sequential()
model.add(Embedding(450000, 1, input_length=7, init='uniform'))
model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
score = model.evaluate(X_test, Y_test, batch_size=16)

# 1
# estimator = KerasClassifier(build_fn=model, nb_epoch=10, batch_size=16, verbose=2)
# print len(Y_train)
# score = cross_val_score(estimator, X_train, Y_train, cv=5)

# 2
# # plot learn curve
# title = "Learning Curves (LSTM)"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# estimator = KerasClassifier(build_fn=model, nb_epoch=10, batch_size=16, verbose=2)
# plc.plot_learning_curve_with_datasize(estimator, title, data, label, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
#
# plt.show()
