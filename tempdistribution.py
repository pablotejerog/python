from math import *
import numpy as np
import matplotlib.pyplot as plt
import time

# Constantes
d = 0.5
Q = 0.6
k = 0.16
H = 0.073
ur = 25

def placa_termica(h: float, rows: int, cols:int, u_izq: float, u_der: float) -> np.array:

    # Trasposición de las variables de entrada previa a la generación de la matriz solución 'u'
    columnas = rows
    rows = cols 

    # Matriz solución traspuesta
    u = np.zeros((rows, columnas))

    # Iteraciones.
    max_iterations = 40000
    n = 0

    # Establecemos los valores de contorno.
    u[0, :] = u_izq
    u[rows-1, :] = u_der
    u_arr = - 15 # Esta condición es en función de la derivada pero es constante así la definimos fuera del bucle.
    u_ab = np.zeros((rows,1), float) # Esta matriz de una columna la utilizaremos para guardar las condiciones de frontera del límite inferior.

    # Variable Epsilon -> Tolerancia
    epsilon = 1e-2

    # Bucle while hasta que la diferencia de las normas esté por debajo de la tolerancia.
    norma = np.linalg.norm(u)
    norma_anterior = norma + 5 # Nos aseguramos de entrar en el bucle while

    # Agoritmo de Gauss-Seidel VECTORIZADO
    while fabs(norma - norma_anterior) > epsilon:
        
        # Copiar la solución anterior
        u_old = np.copy(u)
        norma_anterior = np.linalg.norm(u_old)

        # Condiciones de contorno
        u_ab[:,0]= H/k * (u[:,0] - ur) 
        u[1:-1,columnas-1] = 0.25 * (2*u[1:-1,-2] + u[:-2,columnas-1] + u[2:,columnas-1] - (h*h*Q/(k*d) - 2*h*u_arr))
        u[1:-1,0] = 0.25 * (2*u[1:-1,1] + u[:-2,0] + u[2:,0] - (h*h*Q/(k*d) + 2*h*u_ab[1:-1,0]))
        
        # Puntos interiores
        u[1:-1,1:-1] = 0.25 * (u_old[1:-1, 2:] + u_old[1:-1, :-2] + u_old[2:, 1:-1] + u_old[:-2, 1:-1] -h*h*Q/(k*d))
        
        norma = np.linalg.norm(u)
        n += 1

        if n > max_iterations:
            print(f"El método no converge en {max_iterations}")
            return u, n
    
    return u, n

def main():

    # Medidas de la placa (cm)
    length = 9
    height = 5
    
    # Condiciones de contorno
    u_izq = 20
    u_der = 20

    valores_h = [0.5, 0.25, 0.15, 0.1, 0.05]

    print()

    for h in valores_h:

        # Dimensiones en las que dividimos la tabla en base a 'h'.
        rows = int(height/h)
        cols = int(length/h)

        print(f"-------------{rows}x{cols}----------------")

        t_inicial = time.time()
        
        u, n = placa_termica(h, rows, cols, u_izq, u_der)

        # Transponemos la solución obtenida, ya que se trabaja con la matriz traspuesta
        u = np.transpose(u)

        t_final = time.time()

        if h == 0.1:
            
            print()
            print(f"Temperatura en el centro de la placa = {u[25,45]} grados")
            print(f"Temperatura en el centro del borde inferior = {u[0, 45]} grados")
            print(f"Temperatura en el centro del borde superior = {u[49, 45]} grados")
            print()
        

        print(f"Número de iteraciones: {n}")
        print(f"Tiempo de ejecución (s): {t_final-t_inicial}")
        print()
        
        # Graficación de la solución
        plt.imshow(u, cmap='coolwarm', origin='lower', extent=[0,9,0,5])
        plt.colorbar()
        plt.title(f"Distribución de temperaturas con 'h' = {h}")
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.show()     

main()