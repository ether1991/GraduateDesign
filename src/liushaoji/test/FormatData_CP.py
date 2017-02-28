import pandas
import os
import h5py
import numpy as np

def transformData(inputFilePath):
    input = pandas.read_csv(inputFilePath)
    input.dropna(inplace=True)
    dataSet = input.values

    # X = dataSet[:, 0:7]
    # Y = dataSet[:, 7]*10
    date = dataSet[:, 0]
    x = dataSet[:, 1]

    # trans time format 'HH:mm' to seconds
    def t2s(t):
        h, m = t.strip().split(":")
        return int(h) * 3600 + int(m) * 60

    for i in range(0, len(x)):
        str = date[i]
        length = len(str)
        dataSet[i, 0] = int(str[length-2:length])
        dataSet[i, 1] = t2s(x[i])
        label = int(dataSet[i, 6]*10)
        dataSet[i, 6] =  dataSet[i, 8]
        dataSet[i, 8] = label
    dataSet = map(abs, dataSet)
    # print dataSets
    length = inputFilePath.index('.csv')
    str_pre = inputFilePath[0:length]
    outputPath = str_pre + '_format.csv'
    df = pandas.DataFrame(dataSet)
    df.to_csv(outputPath,header=False, index=False)


filePath = '/Users/liushaoji/PycharmProjects/GraduateDesign/file/'

fileList = os.listdir(filePath)
# print fileList

for fileName in fileList:
    if(fileName.startswith("Day")):
        fInPath = filePath+fileName
        transformData(fInPath)