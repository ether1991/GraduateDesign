import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense

dataframe = pandas.read_csv("/Users/liushaoji/ManageSAEData/PeMS_Data_WithWeek2.csv", delim_whitespace=True, header=0 )
dataset = dataframe.values

print dataset


model = Sequential()
model.add(Dense(14, input_dim=14, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(dataset, dataset )