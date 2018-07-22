import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

data = pd.read_csv('Data KNN.csv')
x, y, z = [], [], []
training_data, test_data, label = [], [], []
training_data_with_label = []
token = 0
tokenx = 0
token_predict = len(data)
half_token_predict = 0.5 * token_predict

x = np.array(data.iloc[:, 0])
y = np.array(data.iloc[:, 1])
z = np.array(data.iloc[:, 2])

while token <= half_token_predict:
    training_data.append([x[token], y[token]])
    label.append(z[token])
    token = token + 1

while tokenx <= half_token_predict:
    training_data_with_label.append([x[tokenx], y[tokenx], z[tokenx]])
    tokenx = tokenx + 1

# (dataset,label,predict,k=?


def fruit_to_color(fruit):
    color_of = {
        'Jeruk': 'y',
        'Pisang': 'g',
        'Durian': 'b'
    }
    return color_of[fruit] if fruit in color_of else 'm'


def K_Nearest_Neighbors(rows, label, prediction, k=5):
    distance = []
    counter = 0
    # print(len(prediction))
    while counter < len(prediction):
        # print(counter)
        predict = prediction[counter]
        for features in rows:
            # print("features:",features, "predict: ",predict)
            euclidian_distance = np.linalg.norm(
                np.array(features) - np.array(predict))
            # print("euclidean distance: ", euclidian_distance)
            distance.append([euclidian_distance, label[counter]])
        # print(rows[counter][1], "    Terhadap:    " ,distance)
        counter = counter + 1
    # print(distance)

    # print(sorted(distance))
    votes = [v[1] for v in sorted(distance)[:k]]
    # print(votes)
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


while token_predict > half_token_predict:
    test_data.append([x[token_predict - 1], y[token_predict - 1]])
    token_predict = token_predict - 1

test_data_individual = [400, 3]

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

print("{} adalah warna buah : {}".format(
    training_data_with_label[2][1], training_data_with_label[2][2]))

for row in training_data_with_label:
    color = fruit_to_color(row[2])
    plt.scatter(row[0], row[1], s=100, color=color)

result = K_Nearest_Neighbors(training_data, label, test_data_individual, k=5)
print("result:  {}".format(result))

color = fruit_to_color(result)
plt.scatter(
    test_data_individual[0], test_data_individual[1], s=100, marker='*', color=color)

plt.show()

# for i in dataset:
#     for ii in i:
#         plt.scatter(ii[0],ii[1],s=100,color=i)

# plt.scatter(plotxtrain,plotytrain)

# test_data untuk group test dan test_data_individualy untuk individual test

# print(data)
