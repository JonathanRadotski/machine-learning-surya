import numpy as np
import matplotlib.pyplot as plt

blob = int(input("Input var x: "))
hidden = int(input("Input var hid: "))
trainblob = np.array([0,0,1,1])
trainhid = np.array([0,1,0,1])
blobar=[blob]
hidar=[hidden]
values = []
i=0

def datapre(blob,hidden):
    for blob in blobar :
        if (blobar[i] > 1 or hidar[i] > 1):
            print("Error Found")
            break


def train(trainblob,trainhid):
    while i < len(trainblob):
        if(trainhid[i] == 1 and trainblob[i] == 1):
            values.append(0)
        if (trainhid[i] == 0 and trainblob[i] == 0):
            values.append(0)
        if (trainhid[i] != trainblob[i]):
            values.append(1)
        i = i+1
    plt.scatter(trainblob,trainhid)
    plt.plot(trainblob,values)
    plt.show()

def test(blobar,hidar):
    if (hidar[i] == 1 and blobar[i] == 1):
        values.append(0)
    if (hidar[i] == 0 and blobar[i] == 0):
        values.append(0)
    if (hidar[i] != blobar[i]):
        values.append(1)
    print(values)

def main():
    train(trainblob,trainhid)

main()