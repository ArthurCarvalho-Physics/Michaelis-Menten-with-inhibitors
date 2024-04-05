# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:43:16 2022

@author: carva
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import expm


start_time = time.time()


# Quantity of substances
Ns = 6
Ne = 3
Ni = 3

# Rate constants
r0 = 10
r1 = 5
r2 = 0
r3 = 10
r4 = 5
r5 = 9
r6 = 4
r7 = 10
r8 = 5
r9 = 0


# Partial and noncompetitive inhibition
def partial(Ns, Ne, Ni):
    config = []
    for i in range(Ns + 1):
        for j in range(Ne + 1):
            for k in range(Ni + 1):
                for l in range(Ns + 1):
                    S = i
                    E = j
                    I = k
                    P = l
                    C1 = Ne - E - Ni + I
                    C3 = Ns - S - P - C1
                    C2 = Ni - I - C3
                
                    if all(var >= 0 for var in (C1, C2, P, E, S, C3, I)) and not (S == 0 and C2 == Ni):
                        config.append([S, E, I, C1, P, C2, C3])                        
    return config


# Determine the quantity of state
config = partial(Ns, Ne, Ni)

# Begins with the state with the initial quantity of substances
config.sort(reverse = True)

# Quantity of states
N = len(config)


# Kronecker Delta
def KD(a, b):
    return 1 if a == b else 0


# Function that determines the mean value of each substance
def media(stateK, H, j):
    x, y = 0, 0
    for i in range(len(stateK)):
        if stateK[i][j] != 0:
            x += stateK[i][j]*H[i][0]
            y += (stateK[i][j]**2)*H[i][0]
    return x, y-x**2


# Quasi-Hamiltonian
def Hamiltonian(sk, sb, r2, r9):
    
    H1 = r0*sk[0]*sk[1]*(KD(sb[0],sk[0]-1)*KD(sb[1],sk[1]-1)*KD(sb[2],sk[2])*KD(sb[3],sk[3]+1)*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]))
    
    H2 = r1*sk[3]*(KD(sb[0],sk[0]+1)*KD(sb[1],sk[1]+1)*KD(sb[2],sk[2])*KD(sb[3],sk[3]-1)*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]))
    
    H3 = r2*sk[3]*(KD(sb[0],sk[0])*KD(sb[1],sk[1]+1)*KD(sb[2],sk[2])*KD(sb[3],sk[3]-1)*KD(sb[4],sk[4]+1)*KD(sb[5],sk[5])*KD(sb[6],sk[6]) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]))

    H4 = r3*sk[1]*sk[0]*sk[2]*(KD(sb[0],sk[0])*KD(sb[1],sk[1]-1)*KD(sb[2],sk[2]-1)*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]+1)*KD(sb[6],sk[6]) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]))
    
    H5 = r4*sk[5]*sk[0]*(KD(sb[0],sk[0])*KD(sb[1],sk[1]+1)*KD(sb[2],sk[2]+1)*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]-1)*KD(sb[6],sk[6]) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]))
    
    H6 = r5*sk[3]*sk[2]*(KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2]-1)*KD(sb[3],sk[3]-1)*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]+1) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]))
    
    H7 = r6*sk[6]*(KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2]+1)*KD(sb[3],sk[3]+1)*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]-1) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]))
    
    H8 = r7*sk[5]*sk[0]*(KD(sb[0],sk[0]-1)*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]-1)*KD(sb[6],sk[6]+1) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]))
    
    H9 = r8*sk[6]*(KD(sb[0],sk[0]+1)*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]+1)*KD(sb[6],sk[6]-1) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]))
    
    H10 = r9*sk[6]*(KD(sb[0],sk[0])*KD(sb[1],sk[1]+1)*KD(sb[2],sk[2]+1)*KD(sb[3],sk[3])*KD(sb[4],sk[4]+1)*KD(sb[5],sk[5])*KD(sb[6],sk[6]-1) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5])*KD(sb[6],sk[6]))    
    
    return H1 + H2 + H3 + H4 + H5 + H6 + H7 + H8 + H9 + H10


# Creates the matrix H:
def matrix(r2, r9):
    m = []
    for j in range(N):
        linha = []    
        stateBra = config[j]     
        for i in range(N):
            stateKet  = config[i]
            H = -Hamiltonian(stateKet, stateBra, r2, r9)
            linha.append(H)
        m.append(linha)
    
    m = np.array(m)
    
    return m


# Lists:
var1_list = []
var2_list = []
det_list = []

inc = 0.05

t = 3

for i in np.arange(0, 2 + inc, inc):
    
    for j in np.arange(0, 2 + inc, inc):
    
        m = matrix(i, j)
        
        P = expm(-m*t).tolist()
            
        p = media(config, P, 4)[0]
        
        det_list.append(float(p))
        var1_list.append(float(i))
        var2_list.append(float(j))


lim = int(np.sqrt(len(var1_list)))

fig, ax = plt.subplots(figsize=(10,8))

x = np.reshape(var1_list,(lim,lim))
y = np.reshape(var2_list,(lim,lim))
z = np.reshape(det_list,(lim,lim))
c = ax.pcolormesh(x, y, z, cmap='inferno', vmin=0, vmax=config[0][0], shading='auto')

ax.axis([x.min(), x.max(), y.min(), y.max()])
ax.set_xlabel(r'$k_{2}$', fontsize=25)
ax.set_ylabel(r'$k_{6}$', fontsize=25)
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

plt.yticks(np.arange(0,2.5,0.5))
plt.ylim(0,2)
plt.xticks(np.arange(0,2.5,0.5))
plt.xlim(0,2)

plt.axline((0, 0.0), slope=1, color="black", linestyle=(0, (5, 5)), linewidth=2)

ax.text(0.62, 0.72, r'$k_{6}=k_{2}$ (No inhibitor)', rotation=45, 
        fontsize=20)
ax.text(0.2, 1.5, r'$k_{6}>k_{2}$ (Activator)', rotation=0, 
        fontsize=20)
ax.text(1.05, 0.5, r'$k_{6}<k_{2}$ (Inhibitor)', rotation=0, 
        fontsize=20)

fig.colorbar(c)

#fig.savefig('Heatmap_partial_product.pdf', bbox_inches='tight',format='pdf', dpi=400)

print("Runtime: ", time.time() - start_time)