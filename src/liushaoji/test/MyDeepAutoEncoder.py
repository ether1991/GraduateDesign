import pandas
from keras.layers import Dense, Input, Dropout
from keras.models import Model

#读入数据
iris_data = pandas.read_csv("/Users/liushaoji/ManageSAEData/iris.txt", header=None )
data = iris_data.values

iris = data[0:, :4]
iris_test = data[23:100, :4]
# print iris
label = data[:, 4]
label_test = data[23:100, 4]
# print label

#定义输入
inputData = Input(shape=(4,))

encode = Dense(5, activation='tanh')(inputData)
encode = Dense(3, activation='tanh')(encode)
decode = Dense(1, activation='sigmoid')(encode)
# decode = Dense(4, activation='sigmoid')(decode)

#定义模型
autoencoder = Model(input=inputData, output=decode)

#编译模型
autoencoder.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#训练模型
autoencoder.fit(iris, label, batch_size=32, nb_epoch=100, verbose=2)

#使用测试集对模型进行检验
# score, acc = autoencoder.evaluate(iris_test, label_test, verbose=2)
# print score
# print acc