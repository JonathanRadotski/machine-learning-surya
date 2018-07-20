import pandas as pd
import numpy as np
from sklearn import tree
import graphviz
from sklearn.metrics import accuracy_score
import matplotlib as plt

xtest=[]
x=[]
y=[]
token=0
#get data
data= pd.read_csv('DataTree.csv')

#Setting Data features (Training features)
xRaw=np.array(data.iloc[:,[1]])
yRaw=np.array(data.iloc[:,[9]])

yRaw[yRaw == 'low'] = 0
yRaw[yRaw == 'medium'] = 1
yRaw[yRaw == 'high'] = 1

data_dict = {0 : [],
             1 : []}
for row in xRaw:
    x.append(row[0])
    y.append(row[0])

while token <= 10:
    if(yRaw[token] == 0):
        data_dict[0].append([x[token],y[token]])
    else:
        data_dict[1].append([x[token],y[token]])
    token = token+1

print("X: ",data_dict[0])

class DecisionTree:
    def __init__(self,visualization = True):
        self.visualization = visualization
        self.colors ={0:'b', 1:'r'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    


'''
#Split data trainset and testset
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=30)

#training classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

#predicting
y_pred = clf.predict(X_test)

#testing accuracy
accuracy = accuracy_score(y_test,y_pred)
print(str(accuracy*100)+"% accuracy")
'''



