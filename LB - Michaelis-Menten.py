

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import expm
from scipy import integrate



start_time = time.time() 


# rate constants
r0 = 0.5      # k1+
r1 = 0.5       # k1-
r2 = 5       # k2
r3 = 0.5       # k3+
r4 = 0.5       # k3-
r5 = 0       # k4+
r6 = 0       # k4-
r7 = 0       # k5+
r8 = 0       # k5-
r9 = 0       # k6


# Quantity of substances
Ne = 1


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



# Creates the matrix H:
def matrix(config, states_number):    
    
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
    
    return m


# Function to calculate linear regression
def linear_fit(x, y):
    s = np.linspace(0, 2, 500)
    a, l = np.polyfit(x, y, 1)
    linear = l + a*s
    s = s.tolist()
    linear = linear.tolist()
    return s, linear



Tm, R, S, Si = [], [], [], []

for i in range(1, 4, 1):
    
    Ni = i
    
    l1, l2, l3, l4 = [], [], [], []

    for s in range(1, 11, 1):
        
        Ns = s
        
        config = []
        
        # Determine the quantity of state according to the type of reaction
        if r3 == 0 and r5 == 0 and r7 == 0 and r9 == 0:
            config = michaelis(Ns, Ne)

        elif r3 != 0 and r5 == 0 and r7 == 0 and r9 == 0:
            config = competitive(Ns, Ne, Ni)

        elif r3 == 0 and r5 != 0 and r7 == 0 and r9 == 0:
            config = uncompetitive(Ns, Ne, Ni)
        
        elif r3 != 0 and r5 != 0 and r7 != 0 and r9 == 0:
            config = partial(Ns, Ne, Ni)
    
        else:
            config = partial(Ns, Ne, Ni)


        # Begins with the state with the initial quantity of substances
        config.sort(reverse = True)
    
        # Quantity of states
        states_number = len(config)

        # building the matrix
        m = matrix(config, states_number)

        T, f = [], []
    
        h = 0.00001
        
        # Calculate of the probability
        for t in np.arange(0, 60 + 0.01, 0.01):
    
            P = -(expm(-m*(t+h)) - expm(-m*t))/h
    
            P = P.tolist()
    
            tm = t*prob(config, P, 4)
            f.append(tm)
            
            T.append(t)
            
            if t >= 1:
                if round(tm, 4) == 0:
                    break

        c = integrate.simps(f, T)
        l1.append(c)
        l2.append(1/c)
        l3.append(Ns)
        l4.append(1/Ns)
                
    Tm.append(l1) 
    R.append(l2)
    S.append(l3)
    Si.append(l4)


colors = ['bo', 'ro', 'go']
labels = [r'$ N_I=1 $', r'$ N_I=2 $', r'$ N_I=3 $']


# Graphic
fig, ax = plt.subplots(figsize=(10, 7), facecolor=(1, 1, 1))


for i in range(len(Tm)):
    # fit linear
    ax.plot(linear_fit(Si[i], Tm[i])[0], linear_fit(Si[i], Tm[i])[1], 'k-', 
            linewidth=2)
    
    # Graphics
    ax.plot(Si[i], Tm[i], colors[i], markersize = 12, markeredgewidth='1',
            markeredgecolor='black', label = labels[i])
    
ax.legend(loc=1, bbox_to_anchor=(.99, .3), fontsize=14, frameon = False, 
          ncol=1, handletextpad=0.2, columnspacing = 1)

plt.ylim(0, 5, 1)
plt.xlim(0, 1, .2)

plt.tick_params(labelsize=15)
plt.xlabel('1/S', fontsize=25)
#plt.ylabel(r'$ \langle t \rangle$', fontsize=25)
plt.ylabel('1/v', rotation=90, fontsize=25)
plt.show()

#fig.savefig('LB - noncompetitive).pdf', bbox_inches='tight',format='pdf', dpi=400)

print('\n\nRuntime:', time.time() - start_time)
