import numpy as np
from numpy.typing import NDArray
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
    return np.sum(A) // 2 # Usamos division entera.

def calcular_P(A: NDArray) -> NDArray:
    # Calcula la matriz con el numero esperado de conexiones entre i y j de A.
    # Basado en configuration model
    K: NDArray = TP1.construir_matriz_grado(A)
    E: int = numero_total_conexiones(A)
    P: NDArray = np.outer(np.diag(K), np.diag(K)) / (2*E)
    return P

def calcula_R(A: NDArray) -> NDArray:
    # La función recibe la matriz de adyacencia A y calcula la matriz de modularidad
    P: NDArray = calcular_P(A)
    R: NDArray = A - P
    return R


def calcula_lambda(L: NDArray, v: NDArray) -> float:
    # Recibe L Matriz laplaciana y v: autovector
    # Devuelve el corte asociado a v.
    s: NDArray = np.sign(v)
    Λ: float = 1/4 * float (s.T @ (L @ s)) #> No usen caracteres raros en el codigo que se puede romper todo
    return Λ


def calcula_Q(R: NDArray, v: NDArray)-> float:
    # Recibe R matriz y v autovector
    # Retorna la modularidad (a menos de un factor 2E)
    s: NDArray = np.sign(v)
    Q: float = float (s.T @ (R @ s))
    return Q

def autovalor(A: NDArray, v: NDArray) -> np.float64:
    # Recibe una matriz A y un vector v, y calcula el autovalor asociado a v.
    l: np.float64 = (v.T @ A @ v) / (v.T @ v) # Coeficiente de Rayleigh. Desambiguo el array con [0].
    return l

def norma_2(v: NDArray) -> np.float64:
    # Recibe un vector y calcula su norma 2.
    return np.sqrt(np.sum(v**2))

def normalizar(v: NDArray) -> NDArray:
    # Recibe un vector y lo normaliza, dividiendo por su norma 2.
    return v / norma_2(v)


def metpot1(A,tol=1e-8,maxrep=np.inf, seed=234) -> tuple[NDArray, float, bool]:
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   # Devuelve la tupla (autovector, autovalor, alcanzoMaxRep?)
   rng = np.random.default_rng(seed)
   v = rng.uniform(-1, 1, A.shape[0]) # Generamos un vector de partida aleatorio, entre -1 y 1
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
    deflA = A - (l1 * np.linalg.outer(v1,v1)) # Sugerencia, usar la función outer de numpy
    return deflA

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf) -> tuple[NDArray, float, bool]:
   # La funcion aplica el método de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeros autovectores y autovalores de A
   # Devuelve la tupla (autovector, autovalor, alcanzoMaxRep?)
   deflA = A - (l1 * np.linalg.outer(v1,v1))
   return metpot1(deflA,tol,maxrep)


def metpotI(A: NDArray, mu: float, tol=1e-8, maxrep=np.inf) -> tuple[NDArray, float, bool]:
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    # A: Matriz
    # mu: Factor de shifting
    # tol: Precision
    # maxrep: Cantidad maxima de repeticiones

    # Aplicamos shifting de autovalores, factorizamos e invertimos la matriz.
    M: NDArray = A + (mu * np.identity(A.shape[0]))
    L, U = TP1.calculaLU(M)
    Mi = TP1.inversa_LU(L,U) # Invertimos M
    
    return metpot1(Mi, tol=tol, maxrep=maxrep) # Usamos el método de la potencia ya implementado.

def metpotI2(A,mu,tol=1e-8,maxrep=np.inf) -> tuple[NDArray, float, bool]:
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el método llegó a converger.
   X = A + (mu * np.identity(A.shape[0])) # Calculamos la matriz A shifteada en mu

   L, U = TP1.calculaLU(X)
   iX = TP1.inversa_LU(L,U) # Invertimos la matriz

   defliX = deflaciona(iX) # La deflacionamos
   v,l,_ =  metpot1(defliX) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_

comunidad = list[int] # Una comunidad es una lista de indices de vertices.
listaComunidades = list[comunidad] | None # Alias del tipo de nombres_s

def filtrar_nombres_signo(nombres_s: listaComunidades, v: NDArray) -> tuple[comunidad, comunidad]:
    return [ni for ni,vi in zip(nombres_s[0],v) if vi>0], [ni for ni,vi in zip(nombres_s[0],v) if vi<0]

def laplaciano_iterativo(A: NDArray, niveles: int, nombres_s: listaComunidades = None) -> listaComunidades:
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = [list(range(A.shape[0]))]
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return nombres_s
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v,_,_ = metpotI2(L, 1) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap = A[v>0,:][:,v>0] # Asociado al signo positivo
        Am = A[v<0,:][:,v<0] # Asociado al signo negativo
        
        # Filtramos los nombres por el signo del autovector
        nombres_pos, nombres_neg = filtrar_nombres_signo(nombres_s, v)

        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[nombres_pos]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[nombres_neg])
                )        

def modularidad_iterativo(A: NDArray, R: NDArray | None = None, nombres_s: listaComunidades = None) -> listaComunidades:
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = [list(range(R.shape[0]))]
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return []
    else:
        v,_,_ = metpot1(R) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return []
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = R[(v>0),:][:,(v>0)] # Parte de R asociada a los valores positivos de v
            Rm = R[(v<0),:][:,(v<0)] # Parte asociada a los valores negativos de v
            vp,_,_ = metpot1(Rp)  # autovector principal de Rp
            vm,_,_ = metpot1(Rm) # autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            nombres_pos, nombres_neg = filtrar_nombres_signo(nombres_s, v)
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([nombres_pos, nombres_neg])
            else:
                # Sino, repetimos para los subniveles
                
                return (
                    modularidad_iterativo(A[(v>0),:][:,(v>0)], Rp, [nombres_pos]) + 
                    modularidad_iterativo(A[(v<0),:][:,(v<0)], Rm, [nombres_neg])
                )
            

def construir_adyacencias_simetricas(D, m):
    # D: matriz de distancias
    # m: Cantidad de links por nodo
    # Retorna una matriz de adyacencia simétrica
    A = TP1.construye_adyacencia(D,m)
    A_simetrica = np.ceil(1/2 * (A + A.T)) # Hacemos simétrica la matriz de adyacencia
    return A_simetrica

def graficar_red_por_particiones_2x2(D, ms: list[int], museos, barrios, laplaciano: bool = True, iteraciones: int = 2, factor_escala = 1e4):
    # D: matriz de distancias
    # ms: Secuencia de cantidad de links por nodo
    # museos y barrios: datos
    # laplaciano: Usar laplaciano iterativo o modularidad para encontrar comunidades.
    # iteraciones: Cantidad de iteraciones a realizar en el laplaciano iterativo
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
        "#98df8a",  # verde claro
        "#ff9896",  # rojo claro
        "#c5b0d5",  # violeta claro
        "#c49c94",  # marrón claro
        "#f7b6d2",  # rosa claro
        "#c7c7c7",  # gris claro
        "#dbdb8d",  # amarillo pálido
        "#9edae5",  # celeste claro
    ]

    if not isinstance(ms, list):
        ms = [ms for i in range(4)]
    
    fig, all_axes = plt.subplots(nrows=2,ncols=2) # Visualización de la red en el mapa
    fig.set_figheight(15)
    fig.set_figwidth(15)  # Aumentamos el tamaño del grafico
    ax = all_axes.flat    # Pasamos la tupla a una lista

    cant_particiones = []

    for i, m in enumerate(ms):
        A = construir_adyacencias_simetricas(D,m)
        G, G_layout = TP1.construir_red_para_visualizar(A, museos)
        particiones: list[list[int]] = []
        if laplaciano:
            particiones = laplaciano_iterativo(A, iteraciones) # type: ignore
        else:
            particiones = modularidad_iterativo(A) # type: ignore
            cant_particiones.append(len(particiones))

        barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax[i]) # Graficamos Los barrios

        nx.draw_networkx(G,G_layout, ax=ax[i],with_labels=False) # Graficamos red

        for j, part in enumerate(particiones):
            nx.draw_networkx_nodes(G, G_layout, nodelist=part, ax=ax[i], node_color=(colores[j]))

        ax[i].text(0.05, 0.95, f'm = {ms[i]}', transform=ax[i].transAxes, fontsize=15,
                verticalalignment='top')

    if laplaciano:
        plt.suptitle(f'Comunidades encontradas con Laplaciano con {iteraciones} corte/s ({2**iteraciones} comunidades)', fontsize=20, y=0.93) #titulo
    else:
        plt.suptitle('Comunidades encontradas con Modularidad', fontsize=20, y=0.93) #titulo
        for k in range(4):
            ax[k].set_title(f'{cant_particiones[k]} comunidades', fontsize=16)

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
    print("Laplaciano iterativo de A_ejemplo:")
    print(laplaciano_iterativo(A_ejemplo, 2))
    print(modularidad_iterativo(A_ejemplo))

    # Leemos el archivo, retenemos aquellos museos que están en CABA, y descartamos aquellos que no tienen latitud y longitud
    museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
    barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')
    D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()
    ms = [3,5,10,50]
    # for i, m in enumerate(ms):
    graficar_red_por_particiones_2x2(D, ms=ms, museos=museos, barrios=barrios, laplaciano=False)
