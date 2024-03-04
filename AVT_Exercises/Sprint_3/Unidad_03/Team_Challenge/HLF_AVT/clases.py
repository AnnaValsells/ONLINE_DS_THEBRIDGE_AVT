import numpy as np
import random
import variables as vs
###

class Tablero:

    def __init__(self,dimens:tuple,carac_agua:str):
        self.dim = dimens
        self.water = carac_agua

    def crear_tablero(self):
        tablero = np.full(self.dim,self.water)
        return tablero

    def el_barco_cabe(self,barco,coordenadas):
        pass

    def choca_con_otro_barco(self,barco,coordenadas):
        pass

    def colocar_barco(self,coordenadas,tamanyo,orientacion):
        pass

    
class Barco:

    def __init__(self, tamanyo, orientacion):
        self.tamanyo = tamanyo
        self.orientacion = orientacion
    
    #

# class Pintor:
#     agua = "〽︎"
#     barco = "⛴︎"
#     disparo_barco = "✴︎"
#     disparo_agua = "▪︎"

#     def __init__(self):

#     def pintar(self, tablero):
#         print(tablero.crear_tablero)
        
# Tres tableros ??
        
    
        