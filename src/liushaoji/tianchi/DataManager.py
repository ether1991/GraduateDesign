import pandas


input = pandas.read_csv('/Users/liushaoji/Downloads/TianChi/dataset/test.txt')
input.dropna(inplace=True)  # drop the NaN records
inputData = input.values
# print inputData


