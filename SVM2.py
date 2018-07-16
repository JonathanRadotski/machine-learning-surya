import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from random import randrange

style.use('ggplot')
ytemp=[]
xtemp=[]
data = pd.read_csv('SVMtrain.csv')
x = np.array(data.iloc[:,[0]])#maxtemp
y = np.array(data.iloc[:,[3]])#Humidity %
token = 15

for row in x:
    xtemp.append(row[0])
for row in y:
    ytemp.append(row[0])

data_dict = {
            1:[],
            -1:[]
            }
#data preprocessing
for row in ytemp:
    if token > 0:
        if (ytemp[row]>60):
            data_dict[1].append([xtemp[row],ytemp[row]])
        if (ytemp[row] <= 60):
            data_dict[-1].append([xtemp[row],ytemp[row]])
    token = token -1

print(data_dict[1])

class Support_Vector_Machine:
    def __init__(self, visualization = True):
        self.visualization = visualization
        self.colors = {1:'b',-1:'r'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    #train
    def fit(self,data):
        self.data = data
        #(||w|| = [w,b])
        opt_dict ={}
        transforms = [[1,1],[-1,1],[1,-1],[-1,-1]]

        #finding the values
        all_data=[]
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        # no need to keep this memory.
        all_data = None

        #stepsize

        step_sizes = [1.0*self.max_feature_value * 0,1,
                      1.0*self.max_feature_value * 0.01,
                      1.0*self.max_feature_value * 0.001,]
        step_sizes.remove(0)
        print(step_sizes)


        b_range_multiple = 5
        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            optimized = False
            print(step)

            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    print("Processing...")
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                if w[0] < 0:
                    optimized = True
                    print("optimized a step.")
                else:
                    # w = (value,value) - (step,step)
                    w = w-step
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2


    def predict(self,features):
        #sign(x.w+b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='+', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

#prediction
xtp=[]
ytp=[]
rand = []
i=0
data_predict = []
pred = pd.read_csv('SVMtest.csv')
xp = np.array(pred.iloc[:,[0]])#maxtemp
yp = np.array(pred.iloc[:,[3]])#Humidity %
tokenpred = 10

for row in xp:
    xtp.append(row[0])

for row in yp:
    ytp.append(row[0])

while i <= tokenpred:
    randomized_row = randrange(0,len(xp))
    rand.append(randomized_row)
    randomized_row = None
    i = i+1
print(rand)


for row in np.arange(0,10,1):
    if tokenpred != 0:
        data_predict.append([xtp[rand[row]],ytp[rand[row]]])
    tokenpred = tokenpred -1
print("PREDICT :", data_predict)

for p in data_predict:
    svm.predict(p)

svm.visualize()