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
from scipy.optimize import curve_fit



start_time = time.time() 


# Rate constants
r0 = 1      #k1+
r1 = 1       #k1-   
r2 = 5       #k2
r3 = 0      #k3+
r4 = 0       #k3-
r5 = 1       #k4+
r6 = 1       #k4-
r7 = 0      #k5+
r8 = 0       #k5-
r9 = 0      #k6



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


# Competitive inhibition
def competitive(Ns, Ne, Ni):
    for i in range(Ns + 1):
        for j in range(Ne + 1):
            for k in range(Ni + 1):
                S = i
                E = j
                I = k
                C1 = Ne - E - Ni + I
                C3 = 0
                C2 = Ni - I
                P = Ns - S - C1
                
                if all(var >= 0 for var in (C1, C2, P, E, S, C3, I)) and not (S == 0 and C2 == Ni):
                    config.append([S, E, I, C1, P, C2, C3])                        
    return config


# Uncompetitive inhibition
def uncompetitive(Ns, Ne, Ni):
    for i in range(Ns + 1):
        for j in range(Ne + 1):
            for k in range(Ni + 1):
                S = i
                E = j
                I = k
                C1 = Ne - E - Ni + I
                C3 = Ni - I
                C2 = 0
                P = Ns - S - C1 - C3
                
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
        if stateK[i][4] == 0:
            x += H[i][0]
    return x

  

# Quasi-Hamiltonian
def Hamiltonian(sk,sb):
    
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


def linear_function(x, a, b):
    return a * x + b

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
    if r3 == 0 and r5 == 0 and r7 == 0 and r9 == 0:
        config = michaelis(Ns, Ne)
        type_reaction = 'michaelis'

    elif r3 != 0 and r5 == 0 and r7 == 0 and r9 == 0:
        config = competitive(Ns, Ne, Ni)
        type_reaction = 'competitive'

    elif r3 == 0 and r5 != 0 and r7 == 0 and r9 == 0:
        config = uncompetitive(Ns, Ne, Ni)
        type_reaction = 'uncompetitive'
        
    elif r3 != 0 and r5 != 0 and r7 != 0 and r9 == 0:
        config = partial(Ns, Ne, Ni)
        type_reaction = 'noncompetitive'
    
    else:
        config = partial(Ns, Ne, Ni)
        type_reaction = 'partial'


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


# This excerpt contains the Ns=5 graph adjustments for each type of inhibition. 
# Some of the adjustments have three timescales, so the code is extensive

# Competitive 
t_competitive = [np.linspace(0, 0.4, 1000), np.linspace(0, 10, 1000)]
y_competitive = [25*t_competitive[0], 1.66866*np.exp(-1.57376*t_competitive[1])]


# Uncompetitive
t_uncompetitive = [np.linspace(0, 0.4, 1000), np.linspace(0, 2, 1000), np.linspace(0.8, 10, 1000)]
y_uncompetitive = [25*t_uncompetitive[0]]

f = F[1]
f_subset, T_subset = f[T.index(T[20]):T.index(T[70])], T[20:70]
params, covariance = curve_fit(linear_function, T_subset, f_subset)
y2 = (0.1+params[1])*np.exp((0.1+params[0])*t_uncompetitive[1])
y3 = 0.225709*np.exp(-0.800362*t_uncompetitive[2])

y_uncompetitive.append(y2)
y_uncompetitive.append(y3)


# Noncompetitive 
t_noncompetitive = [np.linspace(0, 0.4, 1000), np.linspace(0, 2, 1000), np.linspace(0.4, 10, 1000)]
y_noncompetitive = [25*t_noncompetitive[0]]

f = F[2]
f_subset, T_subset = f[T.index(T[20]):T.index(T[70])], T[20:70]
params, covariance = curve_fit(linear_function, T_subset, f_subset)
y2 = (params[1])*np.exp((params[0])*t_noncompetitive[1])
y3 = 0.63904*np.exp(-0.84423*t_noncompetitive[2])

y_noncompetitive.append(y2)
y_noncompetitive.append(y3)


# Partial
t_partial = [np.linspace(0, 0.4, 1000), np.linspace(0, 10, 1000)]
y_partial = [25*t_partial[0], 3.27349*np.exp(-2.41714*t_partial[1])]



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
ax2 = plt.axes([0.25, 0.22, 0.34, 0.32])

plt.ylim(10**-3, 10**1)
plt.xticks(np.arange(0, 16, 0.5))
plt.xlim(-0.1, 3)
plt.ylim(10**-3, 10**1)

# Plot the graph with Ns=5
ax2.semilogy(T, F[1], 'g-', linewidth=5, markersize = 5, markeredgewidth='3', markeredgecolor='g', label = r'$ N_S=5 $')

# Plot the adjustments
if type_reaction == 'competitive':
    for i in range(len(y_competitive)):
        ax2.semilogy(t_competitive[i], y_competitive[i], 'k--', linewidth=3, markersize = 10)

elif type_reaction == 'uncompetitive':
    for i in range(len(y_uncompetitive)):
        ax2.semilogy(t_uncompetitive[i], y_uncompetitive[i], 'k--', linewidth=3, markersize = 10)

elif type_reaction == 'noncompetitive':
    for i in range(len(y_noncompetitive)):
        ax2.semilogy(t_noncompetitive[i], y_noncompetitive[i], 'k--', linewidth=3, markersize = 10)

else:
    for i in range(len(y_partial)):
        ax2.semilogy(t_partial[i], y_partial[i], 'k--', linewidth=3, markersize = 10)


ax2.legend(loc=1, bbox_to_anchor=(.95,.95),fontsize=14,frameon = False, 
          ncol=1, handletextpad=0.2, columnspacing = 1)


plt.tick_params(labelsize=15)
plt.ylabel('f(t)',fontsize=20)
plt.xlabel('t',fontsize=20)

plt.show()

#fig.savefig('FPT_competitive.pdf', bbox_inches='tight',format='pdf', dpi=400)


print('\n\nRun time:', time.time() - start_time)



