# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:38:14 2018

@author: ABHISHEK BHUTRA
"""

#importing the Libraries.....
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the datasets.....
dataset=pd.read_csv("train.csv").values
x=dataset[:,1:]
y=dataset[:,0]

#spliting the dataset into test and train.....
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)'''

#model selection for prediction....
#implementing the K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
c = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski', p=2)
c.fit(x,y)

d=[]
d=x[1]
d.shape=(28,28)

plt.imshow(d)
plt.show()

#checking results for test sets....
test_set=pd.read_csv("test.csv").values

x_test=test_set[:5000,:]

y_pred=c.predict(x_test)

x_test1=test_set[5000:10000,:]
y_pred1=c.predict(x_test1)

x_test2=test_set[10000:15000,:]
y_pred2=c.predict(x_test2)

x_test3=test_set[15000:20000,:]
y_pred3=c.predict(x_test3)

x_test4=test_set[20000:25000,:]
y_pred4=c.predict(x_test4)

x_test5=test_set[25000:28001,:]
y_pred5=c.predict(x_test5)
#precdicting the outcomes....
#y_pred = Classifier.predict([x_test[0]])