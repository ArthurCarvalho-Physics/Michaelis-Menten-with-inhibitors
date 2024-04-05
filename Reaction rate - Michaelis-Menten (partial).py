# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 14:29:31 2023

@author: carva
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import expm
from scipy import integrate



start_time = time.time() 


# rate constants
r0 = 0.5       # k1+
r1 = 0.5       # k1-
r2 = 1         # k2
r3 = 0.5       # k3+
r4 = 0.5       # k3-
r5 = 0.5       # k4+
r6 = 0.5       # k4-
r7 = 0.5       # k5+
r8 = 0.5       # k5-


# Quantity of substances
Ne = 1


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
                
            if all(var >= 0 for var in (C1, C2, P, E, S, C3, I)):
                config.append([S, E, I, C1, P, C2, C3])               
    return config


# Partial and noncompetitive inhibition
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
                        config.append([S, E, I, C1, P, C2, C3])                        
    return config



# Kronecker Delta
def KD(a, b):
    return 1 if a == b else 0


# Function that determines the probabilities of each configuration
def prob(stateK, H, j):
    x = 0
    for i in range(len(stateK)):
        if stateK[i][j] == 0:
            x += H[i][0]
    return x

  


# Quasi-Hamiltonian
def Hamiltonian(sk, sb, r9):
    
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
def matrix(config, states_number, r9):    
    
    m = []

    for j in range(states_number):
        linha = []    
        stateBra = config[j]     
        for i in range(states_number):
            stateKet  = config[i]
            H = -Hamiltonian(stateKet, stateBra, r9)
            linha.append(float(H))
        m.append(linha)

    m = np.array(m)
    
    return m



Tm, S= [], []

I = 0

for r9 in [1, 0.5, 1, 2]:
    
    # lists
    l1, l2 = [], []
            
    for s in range(1, 60, 2):
                
        # all configurations
        config = []
        
        if I == 0:
            config = michaelis(s, Ne)
        
        else:
            config = partial(s, Ne, I)
        
        # Inicia com o maior estado
        config.sort(reverse = True)

        # quantity of states
        states_number = len(config)
                
        # matrix H
        m = matrix(config, states_number, r9)
                
        T, f = [], []
    
        h = 0.00001
        
        
        # Calculate of the probability
        for t in np.arange(0, 120 + 0.01, 0.01):
    
            P = -(expm(-m*(t+h)) - expm(-m*t))/h
    
            P = P.tolist()
    
            tm = t*prob(config, P, 4)
                
            f.append(tm)
            
            T.append(t)
            
            if t >= 1:
                if round(tm, 4) == 0:
                    break

        c = integrate.simps(f, T)

        l1.append(1/c)
        l2.append(s)
                
    Tm.append(l1) 
    S.append(l2)
    
    I = 1



colors = ['k-', 'go-', 'ro-', 'bo-']
linewidths = [3, 5, 5, 5]
labels = [r'$ N_I=0 $', r'$ N_I=1 $', r'$ N_I=1 $', r'$ N_I=1 $']


# Graphic
fig, ax = plt.subplots(figsize=(10, 7), facecolor=(1, 1, 1))

for i in range(len(Tm), 0, -1):
    ax.plot(S[i-1], Tm[i-1], colors[i-1], markersize = 10, linewidth = linewidths[i-1], 
            markeredgewidth='1', markeredgecolor='black', label = labels[i-1])
    
ax.legend(loc=1, bbox_to_anchor=(.2, .99), fontsize=14, frameon = False, 
          ncol=1, handletextpad=0.2, columnspacing = 1)

ax.text(30, max(Tm[1]), r"$ k_6=0.5 $", fontsize=15)
ax.text(30, max(Tm[2]), r"$ k_6=1.0 $", fontsize=15)
ax.text(30, max(Tm[3]), r"$ k_6=2.0 $", fontsize=15)


plt.yticks(np.arange(0, 11, .2))
plt.ylim(0.2, 1.4)
plt.xticks(np.arange(0, 61, 10))
plt.xlim(0, 60)

plt.tick_params(labelsize=15)
plt.xlabel('S', fontsize=25)
plt.ylabel('v', rotation=90, fontsize=25)
plt.show()


#fig.savefig('Reaction_rate_partial.pdf', bbox_inches='tight',format='pdf', dpi=400)


print('\n\nTempo de execução:', time.time() - start_time)
