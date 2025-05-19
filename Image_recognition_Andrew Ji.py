"""
Created on 4/27/2025
This is script for - multivariable calculus used for image recognition.
@author: Andrew Ji
"""

import numpy as np
import random 
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from keras.datasets import mnist

def logisticfunction(z): 
    p = 1.0 / (1.0 + np.exp(-z))
    return p 

def CostFunction(theta, X, y):
    m = len(X)
    Cost = 0
    gradient = np.zeros(len(theta))
    theta=np.array(theta).reshape(len(theta),1)
    # Cost function with Logistic Regression
    Cost=sum(np.array(-np.array(y)*np.log(logisticfunction(X@theta)))-
             np.array((1-np.array(y))*np.log(1-logisticfunction(X@theta))))/m 
    gradient = np.transpose(X)@(logisticfunction(X@theta)-np.array(y))/m
    
  
    return Cost, gradient

# trains multiple logistic regression classifiers
def training(X, y, num_labels):
    m = len(X)
    n = len(X[0,:])
    all_theta = np.zeros((num_labels, n + 1))
    X = np.concatenate((np.ones((m, 1)), X),axis=1)
    for i in range(num_labels):
        initial_theta = np.zeros((n + 1,1))      
        labelofdigit=(np.array(y)==i).astype(int)
        Result = optimize.minimize(CostFunction,initial_theta.flatten(),args=(X, labelofdigit),method='TNC',jac=True,options={'maxiter':100})
        optimal_theta = Result.x
        all_theta[i,:]=np.transpose(optimal_theta)         
    return all_theta

# Predict the label 
def predict(all_theta, X):
    m = len(X)
    y = np.zeros((len(X), 1))
    X = np.concatenate((np.ones((m, 1)), X),axis=1)
    y = np.argmax(logisticfunction(X @ np.transpose(all_theta)),axis=1)
    return y


#import the image data from python  MNIST lib =============
# MNIST(Modified National Institute of Standards and Technology database) is a large collection of handwritten digits.
# It contains samples of handwritten digits from 0 to 9 and has 60,000 and 10,000 training and testing images
(x_train,y_train),(x_test,y_test)=mnist.load_data()
y_train=y_train.reshape(len(y_train),1)
y_test=y_test.reshape(len(y_test),1)

num_labels = 10
# visualize the data ======================================
n=10
fit,ax = plt.subplots(n,n)
for i in range(n):
    for j in range(n):
        k=random.randint(1, len(x_train))-1
        ax[i][j].imshow(x_train[k]) 
plt.show()        

#  Training ===============================================
num_pixel = len(x_train[0,0:])**2
all_theta = training(x_train.reshape(len(x_train),num_pixel), y_train, num_labels)

# Predict =================================================
pred = predict(all_theta, x_train.reshape(len(x_train),num_pixel))
accuracy = np.mean(np.array(pred).reshape(len(pred),1)==y_train)

pred_test = predict(all_theta, x_test.reshape(len(x_test),num_pixel))
accuracy_test = np.mean(np.array(pred_test).reshape(len(pred_test),1)==y_test)

print(accuracy, accuracy_test)






      




