# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 11:12:36 2018

@author: ABHISHEK BHUTRA
"""
#importing the libraries.....
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the datasets......
dataset=pd.read_csv("train.csv")
matrix=dataset.iloc[5,1:].values

#recog=matrix_set[0,1:]

matrix=matrix.reshape(28,28)

plt.imshow(matrix,cmap="gray")
plt.show()