import pandas

input = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/TargetData.csv')
dataSet = input.values

# X = dataSet[:, 0:7]
# Y = dataSet[:, 7]*10
date = dataSet[:, 0]
x = dataSet[:, 1]

# trans time format 'HH:mm' to seconds
def t2s(t):
    h, m = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60

for i in range(0,len(x)):
#   print(t2s(x[i]))
    str = date[i]
    length = len(str)
    dataSet[i, 0] = int(str[length-2:length])
    dataSet[i, 1] = t2s(x[i])
    dataSet[i, 8] = dataSet[i, 8]*10
dataSet = map(abs, dataSet)
# print dataSet

df = pandas.DataFrame(dataSet)
df.to_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/TargetDataFormat.csv')