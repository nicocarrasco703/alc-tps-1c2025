#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return

    for j in range(m-1):
        for i in range(j+1, n):
            Ac[i, j] = Ac[i, j] / Ac[j, j]
            cant_op += 1
            for k in range(j+1, m):
                    Ac[i, k] = Ac[i, k] - Ac[j, k] * Ac[i, j]
                    cant_op += 2
            
    L = np.tril(Ac,-1) + np.eye(A.shape[0]) 
    U = np.triu(Ac)
    
    return Ac, L, U, cant_op

def resolver_sistema_triangular_inferior(L, b):
    m = L.shape[0]
    n = L.shape[1]
    X = np.zeros(n)

    X[0] = b[0] / L[0,0]
    for i in range(1,m):
        for j in range(i):
            # X[i] = b[i] - L[i][]
            



def main():

    A = np.array([[2,1,2,3],
                  [4,3,3,4],
                  [-2,2,-4,-12],
                  [4,1,8,-3]])
    print(A)
    Ac, La, Ua, cantA = elim_gaussiana(A)
    print("Ac", Ac)
    print("La", La)
    print("Ua", Ua)

    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    Bc,L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )



if __name__ == "__main__":
    main()
    
    