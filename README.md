# Michaelis-Menten-with-inhibitors
We performed a study of Michaelis-Menten chemical kinetics with inhibitors using the Fock space approach. This methodology consists of rewriting the master equation as a Schrödinger equation:
$$\frac{\partial \ket{\Psi(t)}}{\partial t} = -H(\alpha^\dagger_1,\alpha_1,...,\alpha^\dagger_k,\alpha_k)\ket{\Psi(t)},$$
with solution:
$$\ket{\Psi(t)} = \exp[-H(\alpha^\dagger_1,\alpha_1,...,\alpha^\dagger_k,\alpha_k)t]\ket{\Psi(0)},$$

The solution determines the probability of occurrence of each system configuration. The H factor is a quasi-Hamiltonian that is written in terms of second quantization operators. For Michaelis-Menten kinetics with inhibitor, H is:

```math
\begin{eqnarray}
    H &=& k_{1+}(c^{\dagger}_1 - e^{\dagger} s^{\dagger})es + k_{1-}(e^{\dagger} s^{\dagger} - c^{\dagger}_1)c_1 + k_2(e^{\dagger}p^{\dagger} - c^{\dagger}_1)c_1 + k_{3+}(c^\dagger_2 s^\dagger - e^\dagger s^\dagger  i^\dagger)eis + k_{3-}(e^\dagger i^\dagger s^\dagger - c^\dagger_2 s^\dagger)c_2s \nonumber\\
    & & + k_{4+}(c_3^\dagger - c_1^\dagger i^\dagger)c_1i + k_{4-}(c_1^\dagger i^\dagger - c_3^\dagger)c_3 + k_{5+}(c_3^\dagger - c_2^\dagger s^\dagger)c_2s + k_{5-}(c_2^\dagger s^\dagger - c_3^\dagger)c_3 +k_6(e^\dagger p^\dagger i^\dagger - c_3^\dagger)c_3,
\end{eqnarray}
```

Each configuration is expressed in terms of kets $\ket{\eta}$ and bra $\bra{\eta}$, in our code we use lists to represent these states, and we use an alternative representation of the quasi-Hamiltonian, writing it in terms of Kronecker deltas. Our approach allowed us to determine the average quantity of substances as a function of time t, the time of first formation of product P, the reaction rate and the Lineweaver-Burk linearization. The codes refer to these results.
