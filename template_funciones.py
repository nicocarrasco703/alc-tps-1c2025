import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy

def matriz_de_distancias(museos):
    # Tomamos museos, lo convertimos al sistema de coordenadas de interés, extraemos su geometría (los puntos del mapa), 
    # calculamos sus distancias a los otros puntos de df, redondeamos (obteniendo distancia en metros), y lo convertimos a un array 2D de numpy
    D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()
    return D

def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

def calculaLU(matriz):
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    m=matriz.shape[0]
    n=matriz.shape[1]
    Ac = matriz.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return

    for j in range(m-1):
        for i in range(j+1, n):
            Ac[i, j] = Ac[i, j] / Ac[j, j]
            for k in range(j+1, m):
                    Ac[i, k] = Ac[i, k] - Ac[j, k] * Ac[i, j]
            
    L = np.tril(Ac,-1) + np.eye(matriz.shape[0]) 
    U = np.triu(Ac)
    
    return L, U

def construir_matriz_grado(M):
    # para cada fila sumo todas sus columnas y lo guardo en la matriz de grado
    K = np.zeros(M.shape) # Inicializo la matriz de grado
    for i in range(M.shape[0]):
        K[i,i] = np.sum(M[i,:])
    return K

def inversa_de_triangular(M):
    # inicialzo matriz identidad
    I = np.eye(M.shape[0])
    return scipy.linalg.solve_triangular(M,I)

def crear_matriz_transiciones(M, Kinv):
    C = np.transpose(M) @ Kinv
    return C

def calcula_matriz_C(A): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    K = construir_matriz_grado(A)
    K_inv = inversa_de_triangular(K) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = crear_matriz_transiciones(A, K_inv) # Calcula C multiplicando Kinv y A
    return C

def calcula_pagerank(A,d):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = C.shape[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
     # vector de unos
    
    M = (1-d) * C
    M = np.eye(N) - M
    M = (N/d) * M
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    
    b = np.ones(N) * d # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def calcular_p(D, m, d):
    # D matriz de distancias, 
    # m: Cantidad de links por nodo
    # d: Factor de dumping
    # Retorna: vector p con scores de page rank normalizados
    A = construye_adyacencia(D,m)
    pr = calcula_pagerank(A, d)# Este va a ser su score Page Rank
    pr = pr/pr.sum() # Normalizamos para que sume 1
    return pr 

def construir_red_para_visualizar(A, museos):
    G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia
    # Construimos un layout a partir de las coordenadas geográficas
    G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}
    return G, G_layout

def graficar_red_p(pr, A, museos, barrios, Nprincipales = 0, factor_escala = 1e4):
    # pr: Vector de scores de page rank normalizados
    # A: matriz de adyacencia
    # museos y barrios: datos
    # Nprincipales: Cantidad de principales
    # factor_escala: Escalamos los nodos 10 mil veces para que sean bien visibles
    
    G, G_layout = construir_red_para_visualizar(A, museos)

    fig, ax = plt.subplots(figsize=(15, 15)) # Visualización de la red en el mapa
    principales = np.argsort(pr)[-Nprincipales:] # Identificamos a los N principales
    labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Nombres para esos nodos
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios

    nx.draw_networkx(G,G_layout,node_size = pr*factor_escala, ax=ax,with_labels=False) # Graficamos red
    nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=6, font_color="k") # Agregamos los nombres

    #titulo
    plt.title('Museos de Buenos Aires', fontsize=20)

    plt.show()

def calcula_matriz_C_continua(D): 
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = ... # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F 
    C = ... # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    for i in range(cantidad_de_visitas-1):
        print(i)
        # Sumamos las matrices de transición para cada cantidad de pasos
    return B