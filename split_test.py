import numpy as np

x_test = np.load('test_x.npy')
y_test = np.load('test_y.npy')

group1 = [0,1,8,9]
group2 = [2,3,4,5,6,7]

xtest1 = []
xtest2 = [] 
ytest1 = []
ytest2 = []


for i in range(len(y_test)):
    if y_test[i][0] in group1:
        ytest1.append(group1.index(y_test[i][0]))
        xtest1.append(x_test[i])
    else:
        ytest2.append(group2.index(y_test[i][0]))
        xtest2.append(x_test[i])
        
xtest1 = np.array(xtest1)
ytest1 = np.array(ytest1)
xtest2 = np.array(xtest2)
ytest2 = np.array(ytest2)
