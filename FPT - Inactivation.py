# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:11:15 2022

@author: carva
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import expm
from scipy import integrate


start_time = time.time() 


# Rate constants
r0 = 1       #k1+
r1 = 1       #k1-   
r2 = 5       #k2
r3 = 1       #k3
r4 = 1       #k4



# All possible states
config = []

# Michaelis-Menten
def michaelis(Ns, Ne):
    for i in range(Ns + 1):
        for j in range(Ne + 1):
            S = i
            E = j
            C1 = Ne - E
            P = Ns - S - C1
            C2 = 0
            C3 = 0
            I = 0
                
            if all(var >= 0 for var in (C1, C2, P, E, S, C3, I)) and not (S == 0 and C2 == Ni):
                config.append([S, E, I, C1, P, C2])               
    return config


# Partial and noncompetitive inactivation
def partial(Ns, Ne, Ni):
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
                        config.append([S, E, I, C1, P, C2+C3])                        
    return config


# Competitive inactivator
def competitive(Ns, Ne, Ni):
    for i in range(Ns + 1):
        for j in range(Ne + 1):
            for k in range(Ni + 1):
                S = i
                E = j
                I = k
                CI = Ni - I
                C1 = Ne - E - CI
                P = Ns - S - C1
                
                if all(var >= 0 for var in (C1, CI, P, E, S, I)) and not (S == 0 and CI == Ni):
                    config.append([S, E, I, C1, P, CI])                        
    return config


# Uncompetitive inactivator
def uncompetitive(Ns, Ne, Ni):
    for i in range(Ns + 1):
        for j in range(Ne + 1):
            for k in range(Ni + 1):
                S = i
                E = j
                I = k
                CI = Ni - I                
                C1 = Ne - E - CI
                P = Ns - S - Ne + E
                
                if all(var >= 0 for var in (C1, CI, P, E, S, I)):
                    config.append([S, E, I, C1, P, CI])                        
    return config



# Kronecker Delta
def KD(a, b):
    return 1 if a == b else 0

# Function that determines the probabilities of each configuration
def prob(stateK, H, j):
    x = 0
    for i in range(len(stateK)):
        if stateK[i][4] == 0:
            x += H[i][0]
    return x

  
# Quasi-Hamiltonian
def Hamiltonian(sk,sb):
    
    H1 = r0*sk[0]*sk[1]*(KD(sb[0],sk[0]-1)*KD(sb[1],sk[1]-1)*KD(sb[2],sk[2])*KD(sb[3],sk[3]+1)*KD(sb[4],sk[4])*KD(sb[5],sk[5]) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]))
    
    H2 = r1*sk[3]*(KD(sb[0],sk[0]+1)*KD(sb[1],sk[1]+1)*KD(sb[2],sk[2])*KD(sb[3],sk[3]-1)*KD(sb[4],sk[4])*KD(sb[5],sk[5]) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]))
    
    H3 = r2*sk[3]*(KD(sb[0],sk[0])*KD(sb[1],sk[1]+1)*KD(sb[2],sk[2])*KD(sb[3],sk[3]-1)*KD(sb[4],sk[4]+1)*KD(sb[5],sk[5]) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]))
    
    H4 = r3*sk[1]*sk[2]*sk[0]*(KD(sb[0],sk[0])*KD(sb[1],sk[1]-1)*KD(sb[2],sk[2]-1)*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]+1) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]))
 
    H5 = r4*sk[3]*sk[2]*(KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2]-1)*KD(sb[3],sk[3]-1)*KD(sb[4],sk[4])*KD(sb[5],sk[5]+1) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]))

    return H1 + H2 + H3 + H4 + H5



# Creates the matrix H:
def matrix(config, states_number):    
    
    m = []

    for j in range(states_number):
        linha = []    
        stateBra = config[j]     
        for i in range(states_number):
            stateKet  = config[i]
            H = -Hamiltonian(stateKet, stateBra)
            linha.append(float(H))
        m.append(linha)
        
    m = np.array(m)
    
    return m



F = []

for l in [1, 5, 10, 20]:
    
    Ni = 1
    Ns = l
    Ne = 1
    
    config = []
    
    # Determine the quantity of state according to the type of reaction
    if r3 == 0 and r4 == 0:
        config = michaelis(Ns, Ne)

    elif r3 != 0 and r4 == 0:
        config = competitive(Ns, Ne, Ni)

    elif r3 == 0 and r4 != 0:
        config = uncompetitive(Ns, Ne, Ni)
    
    else:
        config = partial(Ns, Ne, Ni)

    # Begins with the state with the initial quantity of substances
    config.sort(reverse = True)
    # Quantity of states
    states_number = len(config)

    m = matrix(config, states_number)

    T, f = [], []
    
    h = 0.00001

    # Calculate of the probability
    for t in np.arange(0, 15 + 0.01, 0.01):
    
        P = -(expm(-m*(t+h)) - expm(-m*t))/h
    
        P = P.tolist()
    
        f.append( prob(config, P, 4) )
  
        T.append(t)
    
    F.append(f)



# Normalizes FPT
for i in range(len(F)):
    A = integrate.simps(F[i], T)
    F[i] = [elemento / A for elemento in F[i]]


# Graphic
colors = ['b-', 'g-', 'r-', 'm-']
labels = [r'$ N_S=1 $', r'$ N_S=5 $', r'$ N_S=10 $', r'$ N_S=20 $']

fig, ax = plt.subplots(figsize=(10, 7), facecolor=(1, 1, 1))

plt.ylim(10**-4, 10**1)
plt.xlim(10**-2, 10**1)

for i in range(len(F)):
    ax.loglog(T, F[i], colors[i], linewidth=3, markersize = 5, markeredgewidth='3', markeredgecolor='b', label = labels[i])

ax.legend(loc=1, bbox_to_anchor=(.99,1),fontsize=14,frameon = False, 
          ncol=1, handletextpad=0.2, columnspacing = 1)

plt.tick_params(labelsize=15)
plt.rcParams['legend.fontsize'] = 18
plt.xlabel('t', fontsize=25)
plt.ylabel('f(t)', fontsize=25)



# Inset
ax1 = plt.axes([0.25, 0.22, 0.345, 0.345])

plt.ylim(10**-3, 10**1)
plt.xticks(np.arange(0, 5, 0.4))
plt.xlim(-0.05, 1.6)

for i in range(len(F)):
    ax1.semilogy(T, F[i], colors[i], linewidth=3, markersize = 5, markeredgewidth='3', markeredgecolor='b')

plt.tick_params(labelsize=15)
plt.ylabel('f(t)',fontsize=20)
plt.xlabel('t',fontsize=20)


plt.show()

#fig.savefig('FPT_inactivation_competitive.pdf', bbox_inches='tight',format='pdf', dpi=400)



print('\n\nRun time:', time.time() - start_time)



