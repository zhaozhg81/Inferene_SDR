#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 05:47:22 2023

@author: zhaozhigen
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from Inference_SDR.SIR import SIR
import matplotlib.pyplot as plt
import statistics


def add_one(number):
    return number + 1


def generate_demo_data(n, p, H):
    
    ## p = 10
    ## n = 200
    ## H = 20
    
    m = n // H
    
    beta = np.zeros((p, 1))
    beta[0:3, 0] = np.random.normal(0, 1, 3)
    
    X = np.zeros((n, p))
    
    rho = 0.3
    Sigma = np.eye(p)
    elements = np.power(rho, np.concatenate((np.arange(p-1, -1, -1), np.arange(1, p))))
    for i in range(p):
        Sigma[i, :] = elements[p-i-1:2*p-i-1]
    
    X = np.random.normal(size=(n, p))
    X = np.dot(X, np.linalg.cholesky(Sigma))
    
    Y = np.power(np.dot(X, beta), 3)/2 + np.random.normal(0, 1, n).reshape(n,1)
    
    return(X, Y, beta)
    
n=500
p=10
H=20
m=25


X, Y, beta = generate_demo_data(n,p,H)   

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
sir_object = SIR(X_train, Y_train, X_train.shape[0], p, H, 1)
sir_object.get_direction()

X_train_pred = sir_object.predict_SDR(X_train)
X_test_pred = sir_object.predict_SDR( X_test)


# Create the model
model = Ridge()

# Fit the model to the data
model.fit(X_train_pred, Y_train)

# Predict the labels for new data
Y_test_pred = model.predict(X_test_pred)

MSE = statistics.mean( ( (Y_test_pred-Y_test)*(Y_test_pred-Y_test)  )[:,0] )