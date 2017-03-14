import pandas

input = pandas.read_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/day.csv')
input.dropna(inplace=True)
dataSet = input.values

# X = dataSet[:, 0:7]
# Y = dataSet[:, 7]*10

x = dataSet[:, 0]

# trans time format 'HH:mm' to seconds
def t2s(t):
    h, m = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60

for i in range(0,len(x)):
#   print(t2s(x[i]))
    dataSet[i, 0] = t2s(x[i])
    dataSet[i, 7] = int(dataSet[i, 7]*10)
dataSet = map(abs, dataSet)
# print dataSet

df = pandas.DataFrame(dataSet)
df.to_csv('/Users/liushaoji/PycharmProjects/GraduateDesign/file/dayFormat.csv')