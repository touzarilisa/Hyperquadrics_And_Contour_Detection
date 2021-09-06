# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:26:55 2021

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt
import math

#calcul de l'hyperquadrique
def HQ (x,y,A,B,C,gamma,Nh):    
    hq=0
    for i in range (Nh):
        hq+=(np.abs(A[i]*x+B[i]*y+C[i]))**gamma[i]           
    return hq-1

#tracé des droites englobantes
def droites (x,A,B,C):    
    y1=[]
    y2=[]
    x1=[]
    x2=[]  
    
    if B!=0:  
        for i in range (len(x)):                     
            y1.append(-A*x[i]/B-C/B+1/B) 
            y2.append(-A*x[i]/B-C/B-1/B)
        plt.plot(x,y1,'r')
        plt.plot(x,y2,'b')
        i=0
    elif B==0: 
        for i in range (len(x)):                        
            x1.append(-C/A+1/A)
            x2.append(-C/A-1/A)
        plt.plot(x1,x,'r')
        plt.plot(x2,x,'b')        
    return None 

def psi(x,y,a,b):
    return (a*x+b*y)**4+(x+y)**4-1

#calcul du critère
def J(x,y,a,b):    
    j=0    
    for i in range(len(x)):
        j=j+psi(x[i],y[i],a,b)**2
        
    return j

#calcul de la derivée du critère
def dJ(x,y,a,b):
    s=np.array([[0],[0]])
    for i in range(len(x)):
        s=s+np.array([[(8*x[i]*(a*x[i]+b*y[i])**3)*psi(x[i],y[i],a,b)],[(8*y[i]*(a*x[i]+b*y[i])**3)*psi(x[i],y[i],a,b)]])
    return s

#calcul de la matrice hessienne
def HJ(x,y,a,b):
    s=np.array(([0, 0],[0, 0]))
    for i in range(len(x)):
        s=s+np.array(([32*x[i]**2*(a*x[i]+b*y[i])**6+24*x[i]**2*(a*x[i]+b*y[i])**2*((a*x[i]+b*y[i])**4+(x[i]+y[i])**4-1),32*x[i]*y[i]*(a*x[i]+b*y[i])**6+24*x[i]*y[i]*(a*x[i]+b*y[i])**2*((a*x[i]+b*y[i])**4+(x[i]+y[i])**4-1)],[32*x[i]*y[i]*(a*x[i]+b*y[i])**6+24*x[i]*y[i]*(a*x[i]+b*y[i])**2*((a*x[i]+b*y[i])**4+(x[i]+y[i])**4-1), 32*y[i]**2*(a*x[i]+b*y[i])**6+24*y[i]**2*(a*x[i]+b*y[i])**2*((a*x[i]+b*y[i])**4+(x[i]+y[i])**4-1)]))
    return s

#implémentation de la méthode du gradient
def gradient (x,y,a0,b0,nmax,alpha,precision):
    s=np.array([[a0],[b0]])
    dAB=1
    n=1
            
    while dAB> precision and n<nmax:
        
        derj=dJ(x,y,s[0][n-1],s[1][n-1]) 
        s=np.concatenate((s,[[s[0][n-1]-alpha*derj[0][0]],[s[1][n-1]-alpha*derj[1][0]]]),axis=1)
             
        dAB=alpha*math.sqrt((derj[0][0])**2+(derj[1][0])**2)
        #dAB=alpha*(abs(derj[0][0])+abs(derj[1][0]))
        n=n+1          
    return s,dAB>precision

#implémentation de la méthode de newton
def newton(X0, eps, n,x,y):
    i=0
    delX=np.array([[10],[10]])
    s=X0
    while (i<n) and (abs(np.max(delX))>eps):
        delX = np.linalg.solve(HJ(x,y,X0[0][0],X0[1][0]), -dJ(x,y,X0[0][0],X0[1][0]))
        X0=X0+delX
        s=np.concatenate((s,X0),axis=1)
        i=i+1
    return s