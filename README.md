# Michaelis-Menten-with-inhibitors
We performed a study of Michaelis-Menten chemical kinetics with inhibitors using the Fock space approach. This methodology consists of rewriting the master equation as a Schrödinger equation:

$H\psi = E\psi$

with solution:

\equation

The solution determines the probability of occurrence of each system configuration. The H factor is a quasi-Hamiltonian that is written in terms of second quantization operators.

Each configuration is expressed in terms of kets and bra, in our code we use lists to represent these states, and we use an alternative representation of the quasi-Hamiltonian, writing it in terms of Kronecker deltas.

Our approach allowed us to determine the average quantity of substances as a function of time t, the time of first formation of product P, the reaction rate and the Lineweaver-Burk linearization. The refer to these results.
