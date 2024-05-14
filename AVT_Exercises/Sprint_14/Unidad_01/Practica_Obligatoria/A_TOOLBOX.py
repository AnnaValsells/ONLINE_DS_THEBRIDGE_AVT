import pandas as pd
import numpy as np

def describe_df (data: pd.DataFrame) -> pd.DataFrame: # -> Esto es para decir que te devuelve la función
    '''
    Describe el dtype de cada columna, los valores nulos en %, quantos valores únicos hay en la columna y el % de cardinalidad.

    Argumentos:
    data(pd.DataFrame): DataFrame de Pandas inicial

    Retorna:
    pd.DataFrame: Data inicial transformado con los valores descritos   
    '''

    dic_describe = {
        # Tipo de dtype de cada columna
        'DATA_TYPE' : [data[x].dtype for x in data],
        # % redondeado a 2 de los valores nulos de cada columna
        'MISSINGS (%)' : [round(data[x].isnull().sum()/len(data[x])*100,2) for x in data],
        # Suma de valores únicos de cada columna (dicho de otra manera, la famosa CARDINALIDAD)
        'UNIQUE_VALUES' : [data[x].nunique() for x in data],
        # % de valores unicos por cada columna
        'CARDIN (%)' : [round(data[x].nunique() / len(data[x]) * 100, 2) for x in data]
    }
    
    return pd.DataFrame(dic_describe, index=[x for x in data]).T

###############################################################################################

def _classify(data: pd.DataFrame, key: str,  umbral_categoria:int, umbral_continua:float) -> str: 
    cardi = data[key].nunique() # Calculamos la cardinalidad
    if cardi == 2: # ¿La cardinalidad es igual que dos?
        return "Binaria"
    elif cardi < umbral_categoria: # ¿La cardinalidad es mas pequeña que el número que escogemos para considerar una variable categórica?
        return "Categórica"
    elif cardi/len(data[key])*100 >= umbral_continua: # ¿El % de la cardinalidad es mayor o igual que el número que escogemos para delimitar cuando és Continua o Discreta?
        return "Numérica Continua"
    else:
        return "Numérica Discreta"
        

def tipifica_variable (data:pd.DataFrame, umbral_categoria:int, umbral_continua:float) -> pd.DataFrame:
    '''
    Tipo de variable de cada columna según su cardinalidad.

    Argumentos:
    data(pd.DataFrame): DataFrame inicial
    umbral_categoria(int): Número que escogemos para delimitar a partir de cuanto consideramos que es una variable categorica
    umbral_continua(float): Número que escogemos para delimitar a partir de cuanto una variable numérica es discreta
    
    Retorna:
    pd.DataFrame: Data inicial transformado   
    '''
    # Diccionario con los resultados de las preguntas sobre la cardinalidad
    dic_tip_var = {
        "tipo_sugerido": [_classify(data, key, umbral_categoria, umbral_continua) for key in data]
    }
    # Añadimos un extra, simple print para tener en cuenta si hay valores nulos no tratados en el dataframe
    for x in data:
        hay_nulos = data[x].isnull().sum()
        if hay_nulos != 0:
            print(f'OJO! En la columna "{x}" hay valores nulos.')

    return pd.DataFrame(dic_tip_var, index=[x for x in data])