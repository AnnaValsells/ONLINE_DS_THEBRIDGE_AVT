import numpy as np
import random
import variables as vs

### Conversación inicial para ubicar los barcos del jugador
def rellenar_tablero_jugador(tablero_de_quien):
    coordenadas = input("Escribir coordenadas separadas por una coma: ")
    #---------------------------- Este input es provisional
    coordenadas_leidas = [int(i) for i in coordenadas.split(",")]


# ### DISPARO al agua - o si toca barco X
# def disparo (x,y):
#     if tablero[x,y] == " ":
#         tablero[x,y] = "-"
#     else:
#         tablero[x,y] = "X"