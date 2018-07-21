from __future__ import print_function
import pandas as pd
import numpy as np
import csv

xtest=[]
x,y,z=[],[],[]
token=0
#get data
data= pd.read_csv('DataTree.csv')
token_predict = len(data)

#Setting Data features (Training features)
xRaw=np.array(data.iloc[:,[1]])# last evaluation
yRaw=np.array(data.iloc[:,[9]])# job details
zRaw=np.array(data.iloc[:,[8]])# salary rate

'''
# for svm maybe...
yRaw[yRaw == 'low'] = 0
yRaw[yRaw == 'medium'] = 1
yRaw[yRaw == 'high'] = 1
data_dict = {0 : [],
             1 : [])
'''

training_data = []
for row in xRaw:
    x.append(row[0])
for row in yRaw:
    y.append(row[0])
for row in zRaw:
    z.append(row[0])

while token <= 0.7*(len(data)-1):
    #if(yRaw[token] == 0):
    training_data.append([x[token],z[token],y[token]])
    #else:
     #   data_dict[1].append([x[token],y[token]])
    token = token+1
#training data includes last evaluation, job, and salary rate
header = ['last_evaluation','job','salary rate']

def DataValue(rows,col):
    return set(row[col] for row in rows)

def Class_Count(rows):
    count={}
    for row in rows:
        label = row[2]
        if label not in count:
            count[label] = 0
        count[label] += 1
    return count

def isanumberbro(value):
    confirm = False
    if isinstance(value,int) or isinstance(value,float):
        confirm = True
    '''
    if confirm == False:
        print("False")
    else:
        print("True")
    '''
    return confirm


class Question:
    def __init__(self,column,value):
        self.column = column
        self.value = value

    def match(self,testitpls):
        temp = testitpls[self.column]
        if isanumberbro(temp):
            return temp >= self.value
        else:
            return temp == self.value

    def __repr__(self):
        condition = "=="
        if isanumberbro(self.value):
            condition = ">="
        return 'is %s %s %s ?'%(header[self.column],condition,str(self.value))

#input rows = training_data & question = Question(2(salary_rate),'low')
def classification(rows,question):
    true_rows, false_rows = [],[]
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows,false_rows

#input training_data label
def gini(rows):
    count = Class_Count(rows)
    impurity = 1
    for lbl in count:
        prob_of_lbl = count[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
        print('impurity %s:' %(lbl), impurity)
    return impurity

def informationgain(left,right,current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_the_damn_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])

        for val in values:
            question = Question(col,val)

            true_rows,false_rows = classification(rows,question)
            print(len(true_rows))
            if (len(true_rows)== 0 or len(false_rows) == 0):
                continue
                #dont bother
            gain = informationgain(true_rows,false_rows,current_uncertainty)
            if gain >= best_gain:
                best_gain,best_question = gain, question
    return  best_gain,best_question

class leaf:
    def __init__(self,rows):
        self.prediction = Class_Count(rows)

class Decision_Node:
    def __init__(self,question,true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    gain, question = find_the_damn_split(rows)
    if gain == 0:
        return leaf(rows)
    true_rows,false_rows = classification(rows,question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Decision_Node(question,true_branch,false_branch)

def print_tree(node, spacing =""):
    if isinstance(node, leaf):
        print(spacing + "predict", node.prediction)
        return
    print(spacing + str(node.question))

    # Call this function recursively on the true branch
    print(spacing + '-o True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '-o False:')
    print_tree(node.false_branch, spacing + "  ")


def prediction(row,node):
    if isinstance(node,leaf):
       return node.prediction
    if node.question.match(row):
        return prediction(row, node.true_branch)
    else:
        return prediction(row, node.false_branch)

def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

damn_tree = build_tree(training_data)
print_tree(damn_tree)

testing_data = []

while token_predict > int(round(0.7*(len(data)-1),0)):
    # bla = int(round(0.7*(len(data)-1),0))
    # print(bla)
    # print(x[token_predict-1],y[token_predict-1],z[token_predict-1])
    testing_data.append([x[token_predict-1],z[token_predict-1],y[token_predict-1]])
    token_predict = token_predict-1

for row in testing_data:
    print("Actual: %s with >= %s work evaluation. Predicted: %s" %
           (row[1], row[0], print_leaf(prediction(row,damn_tree))))

'''
print(Class_Count(training_data))
best_gain, best_question = find_the_damn_split(training_data)
print(best_question)
gini(training_data)
'''
'''
gini(training_data)
true_rows, false_rows = classification(training_data, Question(0, float(0.8)))
ig= informationgain(true_rows,false_rows,current_uncertainty=gini(training_data))
print("jumlah class sebelum node pertama:",Class_Count(training_data))
print("Jumlah evaluation yang lebih dari 0.8 class:", Class_Count(true_rows))
print("information gain of [evaluation yang lebih dari 0.8 class]:", ig)
'''
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



