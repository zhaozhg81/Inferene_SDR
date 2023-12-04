#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 07:47:05 2023

@author: zhigenzhao
"""

import numpy as np
from Inference_SDR.SDR import SDR


class SIR(SDR):
    
    def __init__(self, X, Y, n, p, H, numdir):
        SDR.__init__(self, X, Y, n, p)
        self.D = numdir
        self.H = H
        

    def get_direction(self):
        
        m = int(self.n/self.H)
        x_sliced_mean = np.zeros((self.H, self.X.shape[1]))
        
        ORD = np.argsort(self.Y, axis=0 )
        grand_mean = np.mean(self.X, axis=0)
        
        for h in range(self.H):
            x_sliced = self.X[ORD[(h*m):((h+1)*m),0], :]
            x_sliced_mean[h, :] = np.mean(x_sliced, axis=0) - grand_mean
        
        LambdaHat = np.dot(x_sliced_mean.T, x_sliced_mean) / self.H
        eigenvalues, eigenvectors = np.linalg.eig(LambdaHat)
        self.res_sir = eigenvectors[:, :self.D]
        

    
    def predict_SDR(self, X_test):        
        X_pred = np.matmul(X_test, self.res_sir)
        return( X_pred )