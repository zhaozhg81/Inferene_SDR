#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 05:47:22 2023

@author: zhaozhigen
"""

import numpy as np


def add_one(number):
    return number + 1


def generate_demo_data(n, p, H):
    
    p = 10
    n = 200
    
    H = 20
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
    
