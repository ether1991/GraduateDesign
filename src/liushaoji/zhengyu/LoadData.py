import h5py
import pandas

inputData = h5py.File('/Users/liushaoji/PycharmProjects/GraduateDesign/file/NYC14_M16x8_T60_NewEnd.h5', 'r')

# for key in inputData.keys():
    # print inputData[key].shape
    # print inputData[key][:]
    # print '==================================='

for i in range(0, len(inputData['date'])):
    # cha = inputData['data'][i][0] - inputData['data'][i][0]
    print inputData['data'][i][0]
    print '==================================='
    # print inputData['data'][1][0]
