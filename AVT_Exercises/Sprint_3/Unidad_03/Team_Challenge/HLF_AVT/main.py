import sys

import numpy as np
import random
import variables as vs
from clases import *
from funciones import *

###

tablero_jugador = Tablero(vs.dimension,vs.agua)
tablero_ordenador = Tablero(vs.dimension,vs.agua)

print(tablero_jugador.crear_tablero())


# paco = Pintor() # Paco me ayuda a entender el proceso :')

# paco.pintar(tablero_jugador)
# rellenar_tablero_jugador(tablero_jugador)

# Crar dos tableros- TABLERO JUGADOR, TABLERO ORDENADOR
# Colocar Random barcos en ordenador
# Preguntar y colocar barcos jugador
# START --- Conversación de coordenadas ¬ 20 vidas cada uno
# GAME OVER --- Te han hundido TODOS los barcos ¬ 0 vidas jugador
# WIN --- Has hundido TODOS los barcos del ordenador ¬ 0 vidas ordenador



###
__name__=='__main__'
