# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:06:19 2023

@author: carva
"""


import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import expm


start_time = time.time() 

# Quantity initial of substances
Ns = 1
Ne = 1
Ni = 1

# Rate constants
r0 = 10         #k1+
r1 = 5          #k1-   
r2 = 1.5        #k2
r3 = 0.5          #k3
r4 = 0.5        #k4


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


# Determine the quantity of state according to the type of reaction
if r3 == 0 and r4 == 0:
    config = michaelis(Ns, Ne)
    type_reaction = [1, 1, 0, 1, 1, 0]

elif r3 != 0 and r4 == 0:
    config = competitive(Ns, Ne, Ni)
    type_reaction = [1, 1, 1, 1, 1, 1]

elif r3 == 0 and r4 != 0:
    config = uncompetitive(Ns, Ne, Ni)
    type_reaction = [1, 1, 1, 1, 1, 1]
    
else:
    config = partial(Ns, Ne, Ni)
    type_reaction = [1, 1, 1, 1, 1, 1]


# Begins with the state with the initial quantity of substances
config.sort(reverse = True)


# Quantity of states
states_number = len(config)
# Quantity of substances
N = len(config[0])

# Quantity of states:
print("Quantity of states: ",states_number)



# Kronecker Delta
def KD(a, b):
    return 1 if a == b else 0

# Function that determines the mean value of each substance
def mean(config, H, j):
    x, y = 0, 0
    for i in range(len(config)):
        if config[i][j] != 0:
            x += config[i][j]*H[i][0]
            y += (config[i][j]**2)*H[i][0]
    return x, y-x**2


# Quasi-Hamiltonian
def Hamiltonian(sk,sb):
    
    H1 = r0*sk[0]*sk[1]*(KD(sb[0],sk[0]-1)*KD(sb[1],sk[1]-1)*KD(sb[2],sk[2])*KD(sb[3],sk[3]+1)*KD(sb[4],sk[4])*KD(sb[5],sk[5]) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]))
    
    H2 = r1*sk[3]*(KD(sb[0],sk[0]+1)*KD(sb[1],sk[1]+1)*KD(sb[2],sk[2])*KD(sb[3],sk[3]-1)*KD(sb[4],sk[4])*KD(sb[5],sk[5]) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]))
    
    H3 = r2*sk[3]*(KD(sb[0],sk[0])*KD(sb[1],sk[1]+1)*KD(sb[2],sk[2])*KD(sb[3],sk[3]-1)*KD(sb[4],sk[4]+1)*KD(sb[5],sk[5]) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]))
    
    H4 = r3*sk[1]*sk[2]*sk[0]*(KD(sb[0],sk[0])*KD(sb[1],sk[1]-1)*KD(sb[2],sk[2]-1)*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]+1) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]))
 
    H5 = r4*sk[3]*sk[2]*(KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2]-1)*KD(sb[3],sk[3]-1)*KD(sb[4],sk[4])*KD(sb[5],sk[5]+1) - KD(sb[0],sk[0])*KD(sb[1],sk[1])*KD(sb[2],sk[2])*KD(sb[3],sk[3])*KD(sb[4],sk[4])*KD(sb[5],sk[5]))

    return H1 + H2 + H3 + H4 + H5


# Creates the matrix H:
m = []

for j in range(states_number):
     linha = []    
     stateBra = config[j]     
     for i in range(states_number):
         stateKet  = config[i]
         H = -Hamiltonian(stateKet,stateBra)
         linha.append(float(H))
     m.append(linha)

m = np.array(m)


# Lists for substances
y = [[] for i in range(N)]

# Lists for variances
var = [[] for i in range(N)]

# List for time
T = []



# Calculate of the probability
t = 0

while t <= 6:
    
    P = expm(-m*t).tolist()     # probability

    # Appends of averages
    for i in range(N):
        media_result = mean(config, P, i)
        y[i].append(media_result[0])
        var[i].append(media_result[1])
    
    T.append(t)         # appends of time
   
    if t <= 0.1:
        t += 0.001
    
    else:
        t += 0.01
        
    

# Labels of graphics
Label = [r'$\langle S \rangle$', r'$\langle E \rangle$', r'$\langle I \rangle$',
         r'$\langle C_1 \rangle$', r'$\langle P \rangle$', r'$\langle C_I \rangle$']



# Graphics
l = int(config[0][0])

fig, ax = plt.subplots(figsize=(10, 7), facecolor=(1, 1, 1))

for j in range(N):
    if type_reaction[j] == 1:
        ax.plot(T, y[j], linewidth=3, color="C{}".format(j), label=Label[j])

plt.yticks(np.arange(0, 20, 2*l/10))
plt.ylim(0, l+4)
plt.xticks(np.arange(0,17,1))
plt.xlim(-0.05, t)

ax.legend(loc=1, bbox_to_anchor=(.99, 1.0),fontsize=14,frameon = False, 
          ncol=2, handletextpad=0.2, columnspacing = 1)

plt.tick_params(labelsize=15)
plt.xlabel('t',fontsize=20)
plt.ylabel(r'$\langle n_j \rangle$', fontsize=20)



# Inset
ax1 = plt.axes([0.2, 0.53, 0.33, 0.33])
        
for j in range(N):
    ax1.plot(T, y[j], linewidth=3, color="C{}".format(j))

plt.yticks(np.arange(0, 11, 2*l/10))
plt.ylim(0, l)
plt.xticks(np.arange(0, 0.50, 0.05))
plt.xlim(0, 0.2)

plt.tick_params(labelsize=15)
plt.xlabel('t',fontsize=20)
plt.ylabel(r'$\langle n_j \rangle$', fontsize=20, labelpad=-5)

plt.show()
#fig.savefig('Mean_multiple_inactivator_noncompetitive.pdf', bbox_inches='tight',format='pdf', dpi=400)


print('\n\nRuntime:', time.time() - start_time)
