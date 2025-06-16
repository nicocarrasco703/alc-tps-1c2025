import numpy as np
from numpy.typing import NDArray
import scipy
import matplotlib.pyplot as plt
import networkx as nx # Construcción de la red en NetworkX
# Usamos tipado de numpy para que ande bien el linting.
import template_funciones as TP1
import geopandas as gpd 

def calcula_L(A: NDArray) -> NDArray:
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    K: NDArray = TP1.construir_matriz_grado(A)
    # Restamos las adyacencias a los grados.
    L: NDArray = K - A
    return L

def numero_total_conexiones(A: NDArray) -> int:
    # Cuenta cuantas conexiones hay en la matriz de adyacencia A.
    return np.sum(A) / 2

def calcular_P(A: NDArray) -> NDArray:
    # Calcula la matriz con el numero esperado de conexiones entre i y j de A.
    # Basado en configuration model
    K: NDArray = TP1.construir_matriz_grado(A)
    E: int = numero_total_conexiones(A)
    P: NDArray = np.outer(np.diag(K), np.diag(K)) / (2*E)
    return P

def calcula_R(A):
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    P: NDArray = calcular_P(A)
    R: NDArray = A - P
    return R


def calcula_lambda(L: NDArray, v: NDArray) -> float:
    # Recibe L y v y retorna el corte asociado
    s: NDArray = np.sign(v) #todo1: Puedo usar np.sign?
    Λ: float = 1/4 * float (s.T @ (L @ s))
    return Λ


def calcula_Q(R: NDArray, v: NDArray)-> float:
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    s: NDArray = np.sign(v)
    Q: float = float (s.T @ (R @ s))
    return Q

def autovalor(A: NDArray, v: NDArray) -> np.float64:
    l = (v.T @ A @ v) / (v.T @ v)
    return l

def normalizar(v: NDArray):
    return v / np.linalg.norm(v)


def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   v = np.random.uniform(-1, 1, A.shape[0]) # Generamos un vector de partida aleatorio, entre -1 y 1
   v = normalizar(v) # Lo normalizamos
   v1 = A @ v # Aplicamos la matriz una vez
   v1 = normalizar(v1) # normalizamos
   l = autovalor(A, v) # Calculamos el autovalor estimado
   l1 = autovalor(A, v1) # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = A @ v1 # Calculo nuevo v1
      v1 = normalizar(v1) # Normalizo
      l1 = autovalor(A, v1) # Calculo autovalor
      nrep += 1 # Un pasito mas
      if not nrep < maxrep:
        print('MaxRep alcanzado')
   l = autovalor(A, v1) # Calculamos el autovalor final
   return v1,l,nrep<maxrep

def deflaciona(A: NDArray, tol=1e-8, maxrep=np.inf) -> NDArray:
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - (l1 * np.linalg.outer(v1,v1)) # Sugerencia, usar la funcion outer de numpy
    return deflA

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):# todo2: por que tiene parametro v1, l2??
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A}
   deflA = A - (l1 * np.linalg.outer(v1,v1))
   return metpot1(deflA,tol,maxrep)


def metpotI(A: NDArray, mu, tol=1e-8, maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    # todo: Tengo la duda de que el codigo del template tenia: return metpot1(...,tol=tol,maxrep=maxrep)
    M: NDArray = A + (mu * np.identity(A.shape[0]))
    L, U = TP1.calculaLU(M)

    v = np.random.uniform(-1, 1, M.shape[0]) # Generamos un vector de partida aleatorio, entre -1 y 1
    v = normalizar(v) # Lo normalizamos

    # En vez de multiplicar la matriz, v1 = M v, resolvemos el sistema
    v1 = resolverLUinversa(L, U, v)
    v1 = normalizar(v1)
    l = autovalor(M, v) # Estimamos el autovalor, usando la matriz M original.
    l1 = autovalor(M, v1) # Calculamos los primeros pasos
    nrep = 0 # Contador
    while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
        v = v1 # actualizamos v y repetimos
        l = l1
        v1 = resolverLUinversa(L, U, v1)
        v1 = normalizar(v1) # Normalizo
        l1 = autovalor(M, v1) # Calculo autovalor
        nrep += 1 # Un pasito mas
        if not nrep < maxrep:
            print('MaxRep alcanzado')
    l = autovalor(M, v1) # Calculamos el autovalor final
    return v1,l,nrep<maxrep

def resolverLUinversa(L: NDArray, U:NDArray, v:NDArray) -> NDArray:
    # Resuelve el sistema: LU x = v. 
    # Aprovechando que la matriz esta factorizada para no tener que invertirla
    Ly = scipy.linalg.solve_triangular(L,v,lower=True)
    x = scipy.linalg.solve_triangular(U,Ly)
    return x

def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A + (mu * np.identity(A.shape[0])) # Calculamos la matriz A shifteada en mu

   L, U = TP1.calculaLU(X)
   iX = TP1.inversa_LU(L,U) # Invertimos la matriz

   defliX = deflaciona(iX) # La deflacionamos
   v,l,_ =  metpot1(defliX) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_


def laplaciano_iterativo(A: NDArray, niveles: int, nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v,l,_ = metpotI2(L, 1) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap = A[v>0,:][:,v>0] # Asociado al signo positivo
        Am = A[v<0,:][:,v<0] # Asociado al signo negativo
        
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )        

def modularidad_iterativo(A: NDArray, R=None, nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return []
    else:
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return []
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = R[(v>0),:][:,(v>0)] # Parte de R asociada a los valores positivos de v
            Rm = R[(v<0),:][:,(v<0)] # Parte asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm) # autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return (
                    modularidad_iterativo(A[(v>0),:][:,(v>0)], Rp, [ni for ni,vi in zip(nombres_s,v) if vi>0]) + 
                    modularidad_iterativo(A[(v<0),:][:,(v<0)], Rm, [ni for ni,vi in zip(nombres_s,v) if vi<0])
                )
            

def construir_adyacencias_simetricas(D, m):
    A = TP1.construye_adyacencia(D,m)
    A_simetrica = np.ceil(1/2 * (A + A.T))
    return A_simetrica

def graficar_red_por_particiones_2x2(D, ms: list[int], museos, barrios, laplaciano: bool = True, iteraciones: int = 2, factor_escala = 1e4):
    # D: matriz de distancias
    # ms: Secuencia de cantidad de links por nodo
    # museos y barrios: datos
    # laplaciona: Usar laplaciano iterativo o modularidad para encontrar comunidades.
    # factor_escala: Escalamos los nodos 10 mil veces para que sean bien visibles
    # Retorna: Un gráfico de todas las redes de museos particionados
    colores = [
    "#1f77b4",  # azul
    "#ff7f0e",  # naranja
    "#2ca02c",  # verde
    "#d62728",  # rojo
    "#9467bd",  # violeta
    "#8c564b",  # marrón
    "#e377c2",  # rosa
    "#7f7f7f",  # gris
    "#bcbd22",  # verde lima
    "#17becf",  # celeste
    "#aec7e8",  # azul claro
    "#ffbb78",  # naranja claro
    ]

    if not isinstance(ms, list):
        ms = [ms for i in range(4)]
    
    fig, all_axes = plt.subplots(nrows=2,ncols=2) # Visualización de la red en el mapa
    fig.set_figheight(15)
    fig.set_figwidth(15)  # Aumentamos el tamaño del grafico
    ax = all_axes.flat    # Pasamos la tupla a una lista

    for i, m in enumerate(ms):
        A = construir_adyacencias_simetricas(D,m)
        G, G_layout = TP1.construir_red_para_visualizar(A, museos)
        particiones: list[list[int]] = []
        if laplaciano:
            particiones = laplaciano_iterativo(A, iteraciones) # type: ignore
        else:
            particiones = modularidad_iterativo(A) # type: ignore
        print(len(particiones))
        barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax[i]) # Graficamos Los barrios

        nx.draw_networkx(G,G_layout, ax=ax[i],with_labels=False) # Graficamos red

        for j, part in enumerate(particiones):
            nx.draw_networkx_nodes(G, G_layout, nodelist=part, ax=ax[i], node_color=(colores[j % 12])) # todo: corregir cantidad de colores

        ax[i].text(0.05, 0.95, f'm = {ms[i]}', transform=ax[i].transAxes, fontsize=15,
                verticalalignment='top')

    if laplaciano:
        plt.suptitle('Comunidades encontradas con Laplaciano', fontsize=20) #titulo
    else:
        plt.suptitle('Comunidades encontradas con Modularidad', fontsize=20) #titulo

    plt.show()


if __name__ == "__main__":
    np.random.seed(123)
    print("Ejercicio 3 test...\n")
    # Matriz A de ejemplo
    A_ejemplo = np.array(
        [
            [0, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 0],
        ]
    )

    print("Calcular L de A_ejemplo: \n\n")
    print(calcula_L(A_ejemplo))
    print("\nCalcular R de A_ejemplo: \n\n")
    print(calcula_R(A_ejemplo))
    s = np.array([1,1,1,1,-1,-1,-1,-1]) # "autovector v"
    print(calcula_lambda(calcula_R(A_ejemplo), s))
    print(calcula_Q(A_ejemplo, s))

    L = calcula_L(A_ejemplo)
    autovals, autovecs = np.linalg.eig(L)
    autovals, indices = np.sort(autovals)[::-1], np.argsort(autovals)[::-1]
    autovecs = autovecs[:, indices]
    print("\nAutovalores de L: \n")
    print(autovals)
    print("\n")

    # autovalor mas chico
    print("Autovalor mas chico de L sumando mu: ", metpotI(L, 2)[1])
    # segundo autovalor mas chico
    print("Segundo autovalor mas chico de L: ", metpotI2(L, 2)[1])
    print(laplaciano_iterativo(A_ejemplo, 2, ["A","B","C","D","E","F","G","H"]))

    print(modularidad_iterativo(A_ejemplo, None, ["A","B","C","D","E","F","G","H"]))

    # Leemos el archivo, retenemos aquellos museos que están en CABA, y descartamos aquellos que no tienen latitud y longitud
    museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
    barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')
    D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()
    ms = [3,5,10,50]
    # for i, m in enumerate(ms):
    graficar_red_por_particiones_2x2(D, ms=ms, museos=museos, barrios=barrios, laplaciano=False)
