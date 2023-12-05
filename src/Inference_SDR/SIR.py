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
        for d in range(self.D):
            self.res_sir[:,d] = self.res_sir[:,d]/np.dot( self.res_sir[:,d], self.res_sir[:,d])s

    def sir_direction(X,Y,n, p, H,numdir):
        
        m = int(n/H)
        x_sliced_mean = np.zeros((H, X.shape[1]))
        
        ORD = np.argsort(Y, axis=0 )
        grand_mean = np.mean(X, axis=0)
        
        for h in range(H):
            x_sliced = X[ORD[(h*m):((h+1)*m),0], :]
            x_sliced_mean[h, :] = np.mean(x_sliced, axis=0) - grand_mean
        
        LambdaHat = np.dot(x_sliced_mean.T, x_sliced_mean) /H
        
        eigenvalues, eigenvectors = np.linalg.eig(LambdaHat)
        res_sir = eigenvectors[:, :D]
        for d in range(D):
            res_sir[:,d] = res_sir[:,d]/np.dot( res_sir[:,d], res_sir[:,d])s

    
    def predict_SDR(self, X_test):        
        X_pred = np.matmul(X_test, self.res_sir)
        return( X_pred )
    
    
def Lasso_SIR(X,Y,n,p,H,numdir):
    screening = False 
    
    ORD = np.argsort(Y, axis=0)[:,0]
    X_tran = X[ORD, :]
    Y_tran = Y[ORD]
    
    ms = np.zeros(n)
    m = n // H
    c = n % H
    M = np.zeros((H, n))
    if c == 0:
        M = np.kron(np.eye(H), np.ones((1, m))) / m
    else:
        for i in range(1, c+1):
            M[i-1, (m+1)*(i-1): (m+1)*i] = 1 / (m+1)
            ms[(m+1)*(i-1): (m+1)*i] = m
        for i in range(c+1, H+1):
            M[i-1, (m+1)*c + (i-c-1)*m: (m+1)*c + (i-c)*m] = 1 / m
            ms[(m+1)*c + (i-c-1)*m: (m+1)*c + (i-c)*m] = m - 1
    
    if screening:
        x_sliced_mean = np.matmul(M, X_tran)
        sliced_variance = np.apply_along_axis(np.var, 0, x_sliced_mean)
        keep_ind = np.argsort(sliced_variance)[::-1][:n]
    else:
        keep_ind = np.arange(0, p)
    
    X_tran = X_tran[:, keep_ind]
    X_H = np.zeros((H, X.shape[1]))
    grand_mean = np.mean(X, axis=0)
    X_stand_ord = X_tran - np.kron( grand_mean.reshape(1, -1), np.ones((X.shape[0], 1)) )
    X_H = np.matmul( M , X_stand_ord )
    
    lambda_H = np.dot(X_H.T, X_H)/H
    eigenvalues, eigenvectors = np.linalg.eig(lambda_H)
    
    
    beta_hat = np.zeros((p, no_dim))
    Y_tilde = np.zeros((n, no_dim))

    
    for ii in range(0, no_dim):
        Y_tilde[:, ii] = m* np.matmul(np.matmul(np.matmul(M.T, M), X_stand_ord), eigenvectors[ii,:]) / eigenvalues[ii] 
    

