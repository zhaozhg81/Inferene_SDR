#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 07:47:05 2023

@author: zhigenzhao
"""

import numpy as np

def SIR(X, Y, p, m, n, H, numdir):
    x_sliced_mean = np.zeros((H, X.shape[1]))
    
    ORD = np.argsort(Y, axis=0 )
    grand_mean = np.mean(X, axis=0)
    
    for h in range(H):
        x_sliced = X[ORD[(h*m):((h+1)*m),0], :]
        x_sliced_mean[h, :] = np.mean(x_sliced, axis=0) - grand_mean
    
    LambdaHat = np.dot(x_sliced_mean.T, x_sliced_mean) / H
    eigenvalues, eigenvectors = np.linalg.eig(LambdaHat)
    res_sir = eigenvectors[:, :numdir]
    
    return res_sir

