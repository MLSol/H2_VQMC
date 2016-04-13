# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:59:06 2016

@author: Martijn Sol
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:28:43 2016

@author: Martijn Sol
"""

#VQMC to find ground state energy of Helium

#from numba import jit
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
import time

from math import sqrt, exp

N = 400         # Number of walkers
s = 1.4         
beta = 0.58
a = s / (lambertw(s / exp(s)) + s)
a = a.real
MCsteps = 30000   # Number of Monte Carlo steps

dr = 0.5       #maximum step size

# Initialize postions of walkers
def initialize(N):
    r = 3*(np.random.random((N,2,3)) - 0.5)
    return r

# Calculate wave function
def psi(rElectron1, rElectron2, s, beta, a):
    rL1 = 0
    rL2 = 0
    rR1 = 0
    rR2 = 0    
    r12 = 0
    for d in range(3):
        if d == 0:
            rL1 += (rElectron1[0,d] + s/2) * (rElectron1[0,d] + s/2)
            rL2 += (rElectron2[0,d] + s/2) * (rElectron2[0,d] + s/2)
            rR1 += (rElectron1[0,d] - s/2) * (rElectron1[0,d] - s/2)
            rR2 += (rElectron2[0,d] - s/2) * (rElectron2[0,d] - s/2)
        else:
            rL1 += rElectron1[0,d] * rElectron1[0,d]
            rL2 += rElectron2[0,d] * rElectron2[0,d]
            rR1 += rElectron1[0,d] * rElectron1[0,d]
            rR2 += rElectron2[0,d] * rElectron2[0,d]
        
        r12 += (rElectron1[0,d] - rElectron2[0,d]) * (rElectron1[0,d] - rElectron2[0,d])
    rL1 = sqrt(rL1)
    rL2 = sqrt(rL2)
    rR1 = sqrt(rR1)
    rR2 = sqrt(rR2)
    r12 = sqrt(r12)
    phi1 = exp(-rL1 / a) + exp(-rR1 / a)
    phi2 = exp(-rL2 / a) + exp(-rR2 / a)
    phi12 = exp(r12 / (2 * (1 + beta *r12)))
    psi = phi1 * phi2 * phi12
#    print(psi)
    return psi


# Caluculate local energy
def eLocal(rElectron1, rElectron2, s, beta, a):
    r1L = 0
    r2L = 0
    r1R = 0
    r2R = 0    
    r12 = 0
    for d in range(3):
        if d == 0:
            r1L += (rElectron1[0,d] + s/2) * (rElectron1[0,d] + s/2)
            r2L += (rElectron2[0,d] + s/2) * (rElectron2[0,d] + s/2)
            r1R += (rElectron1[0,d] - s/2) * (rElectron1[0,d] - s/2)
            r2R += (rElectron2[0,d] - s/2) * (rElectron2[0,d] - s/2)
        else:
            r1L += rElectron1[0,d] * rElectron1[0,d]
            r2L += rElectron2[0,d] * rElectron2[0,d]
            r1R += rElectron1[0,d] * rElectron1[0,d]
            r2R += rElectron2[0,d] * rElectron2[0,d]
        
        r12 += (rElectron1[0,d] - rElectron2[0,d]) * (rElectron1[0,d] - rElectron2[0,d])
    r1L = sqrt(r1L)
    r2L = sqrt(r2L)
    r1R = sqrt(r1R)
    r2R = sqrt(r2R)
    r12 = sqrt(r12)
    phi1L = exp(-r1L / a)
    phi2L = exp(-r2L / a)
    phi1R = exp(-r1R / a)
    phi2R = exp(-r2R / a)
    phi1 = phi1L + phi1R
    phi2 = phi2L + phi2R
    
    dotProd = 0
    for d in range(3):
        term1 = 0
        if d == 0:        
            term1 += phi1L * (rElectron1[0,d] + s/2) / (phi1 * r1L)
            term1 += phi1R * (rElectron1[0,d] - s/2) / (phi1 * r1R)
            term1 -= phi2L * (rElectron2[0,d] + s/2) / (phi2 * r2L)
            term1 -= phi2R * (rElectron2[0,d] - s/2) / (phi2 * r2R)
        else:
            term1 += phi1L * rElectron1[0,d] / (phi1 * r1L)
            term1 += phi1R * rElectron1[0,d] / (phi1 * r1R)
            term1 -= phi2L * rElectron2[0,d] / (phi2 * r2L)
            term1 -= phi2R * rElectron2[0,d] / (phi2 * r2R)
        dotProd += term1 * (rElectron1[0,d] - rElectron2[0,d]) / r12
        
    el = -1 / (a * a)
    el += (phi1L / r1L + phi1R / r1R) / (a * phi1)
    el += (phi2L / r2L + phi2R / r2R) / (a * phi2)
    el += -(1/r1L + 1/r1R + 1/r2L + 1/r2R) + 1/r12
    el -= ((4 * beta + 1) * r12 + 4) / (4 * (1 + beta * r12)**4 * r12)
    el += dotProd / (2 * a * (1 + beta * r12)**2)
    
    mterm = (-r12 ** 2) / (2 * (1 + beta * r12) ** 2)
    elProd = el * mterm
    
#    print(el)    
    return el, mterm, elProd
    
# Performs a Monte Carlo step
def MCstep(r, dr, eSum, eSqdSum, mtermSum, eProdSum, s, beta, a):
    for i in range(r.shape[0]):
        rElectron1 = r[i,0:1,0:3]
        rTrial1 = rElectron1 + dr * (2 * np.random.random((1,3)) - 1)     
        rElectron2 = r[i,1:2,0:3]
        rTrial2 = rElectron2 + dr * (2 * np.random.random((1,3)) - 1)
    
        p = psi(rTrial1, rTrial2, s, beta, a) / psi(rElectron1, rElectron2, s, beta, a)
        p = p * p
        if p > np.random.random():
            r[i,0,:] = rTrial1
            r[i,1,:] = rTrial2
            rElectron1 = rTrial1
            rElectron2 = rTrial2

        [e, mterm, eProd] = eLocal(rElectron1, rElectron2, s, beta, a)
        
        eSum += e
        mtermSum += mterm
        eProdSum += eProd
        eSqdSum += e*e
    return r, eSum, eSqdSum, mtermSum, eProdSum

#PROGAMM 
#t1 = time.clock()
for q in range(1):
    r = initialize(N)
    eSum = 0
    eSqdSum = 0
    mtermSum = 0
    eProdSum = 0
    
    print('beta =', beta)
    for j in range(int(0.15 * MCsteps)):
        [r, eSum, eSqdSum, mtermSum, eProdSum] = MCstep(r, dr, eSum, eSqdSum, mtermSum, eProdSum, s, beta, a)
        
    eSum = 0
    eSqdSum = 0
    mtermSum = 0
    eProdSum = 0
    for j in range(MCsteps):
        #    t1 = time.clock()
        [r, eSum, eSqdSum, mtermSum, eProdSum] = MCstep(r, dr, eSum, eSqdSum, mtermSum, eProdSum, s, beta, a)
        
#        print(time.clock() - t1)
#        print(j)
    eAverage = eSum / (N * MCsteps)
    eVar = eSqdSum / (N * MCsteps) - eAverage * eAverage
    error = sqrt(eVar) / sqrt(N * MCsteps)
    print(eAverage + 1/s, eVar)
    eProdAve = eProdSum / (N * MCsteps)
    mtermAve = mtermSum / (N * MCsteps)
    dEdbeta = 2 * (eProdAve - eAverage * mtermAve)
    #print(beta - dEdbeta)
    beta -= dEdbeta
    print('time passed =', time.clock() - t1)










