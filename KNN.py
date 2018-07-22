import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

data = pd.read_csv('Data KNN.csv')
xRaw,yRaw, zRaw = [],[],[]
x,y,z = [],[],[]
training_data,test_data,label = [],[],[]
training_data_with_label = []
token= 0
tokenx=0
tokenpredict = len(data)
xRaw= np.array(data.iloc[:,[0]])
yRaw = np.array(data.iloc[:,[2]])
zRaw = np.array(data.iloc[:,[1]])

for row in xRaw:
    x.append(row[0])
for row in yRaw:
    y.append(row[0])
for row in zRaw:
    z.append(row[0])

while token <= 0.5*(len(data)):
    training_data.append([x[token],z[token]])
    label.append(y[token])
    token = token+1

while tokenx <= 0.5*(len(data)):
    training_data_with_label.append([x[tokenx],z[tokenx],y[tokenx]])
    tokenx = tokenx+1

# (dataset,label,predict,k=?
def K_Nearest_Neighbors(rows,label,prediction,k=5):
    distance = []
    counter = 0
    # print(len(prediction))
    while counter < len(prediction):
        # print(counter)
        predict = prediction[counter]
        for row in rows:
            # print(row)
            for features in rows:
                 # print("features:",features, "predict: ",predict)
                euclidian_distance = np.linalg.norm(np.array(features)- np.array(predict))
                # print("euclidean distance: ", euclidian_distance)
                distance.append([euclidian_distance,label[counter]])
        # print(rows[counter][1], "    Terhadap:    " ,distance)
        counter = counter+1
    # print(distance)

    # print(sorted(distance))
    votes = [v[1] for v in sorted(distance)[:k]]
    # print(votes)
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


while tokenpredict > 0.5*(len(data)):
    test_data.append([x[tokenpredict-1],z[tokenpredict-1]])
    tokenpredict = tokenpredict-1

test_data_individual = [400,3]


# print(training_data)
# buah_dict={'Jeruk': [],
#            'Pisang': [],
#            'Durian': []}
# for row in training_data:
#     if training_data_with_label[0][2] == 'Jeruk' or training_data_with_label[0][2] == 'jeruk' :
#         buah_dict['Jeruk'].append(training_data)
#     if training_data_with_label[0][2] == 'Pisang':
#         buah_dict['Pisang'].append(training_data)
#     if training_data_with_label[0][2] == 'Durian':
#         buah_dict['Durian'].append(training_data)
#
# print(buah_dict)

print(training_data_with_label[2][1], "  adalah warna buah :  ", training_data_with_label[2][2])


for row in training_data_with_label:
    if (row[2] == 'Jeruk'):
        color = 'y'
    if (row[2] == 'Pisang'):
        color = 'g'
    if (row[2] == 'Durian'):
        color = 'b'
    plt.scatter(row[0],row[1],s=100,color=color)

result = K_Nearest_Neighbors(training_data,label,test_data_individual, k =5)
print("result:  ", result)
if (result == 'Jeruk'):
    color = 'y'
if (result == 'Pisang'):
    color = 'g'
if (result == 'Durian'):
    color = 'b'

plt.scatter(test_data_individual[0],test_data_individual[1],s=100,marker='*',color=color)

plt.show()


# for i in dataset:
#     for ii in i:
#         plt.scatter(ii[0],ii[1],s=100,color=i)

# plt.scatter(plotxtrain,plotytrain)

# test_data untuk group test dan test_data_individualy untuk individual test

# print(data)