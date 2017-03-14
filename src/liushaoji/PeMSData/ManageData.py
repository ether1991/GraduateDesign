import pandas
import time
dataSet = pandas.read_csv("/Users/liushaoji/PycharmProjects/GraduateDesign/file/day.csv")

data = dataSet.values

line0_Max = data[0:, 0]
for i in range(0 , len(line0_Max)):
    data[i, 0] = time.mktime(time.strptime('2008-01-01 '+line0_Max[i]+':00', "%Y-%m-%d %H:%M:%S"))

data[0:, 3] = (-1)*data[0:, 3]
line0_Max = max(data[0:, 0])
line1_Max = float(max(data[0:, 1]))
line2_Max = max(data[0:, 2])
line3_Max = max(data[0:, 3])
line4_Max = max(data[0:, 4])
line5_Max = max(data[0:, 5])
line6_Max = max(data[0:, 6])
data[0:, 0] = data[0:, 0]/line0_Max
data[0:, 1] = data[0:, 1]/line1_Max
data[0:, 2] = data[0:, 2]/line2_Max
data[0:, 3] = data[0:, 3]/line3_Max
data[0:, 4] = data[0:, 4]/line4_Max
data[0:, 5] = data[0:, 5]/line5_Max
data[0:, 6] = data[0:, 6]/line6_Max
data[0:, 7] = data[0:, 7]*10

print data[0]

save = pandas.DataFrame(data)
save.to_csv("/Users/liushaoji/PycharmProjects/GraduateDesign/file/day001.csv", ",", header=False, index=False)