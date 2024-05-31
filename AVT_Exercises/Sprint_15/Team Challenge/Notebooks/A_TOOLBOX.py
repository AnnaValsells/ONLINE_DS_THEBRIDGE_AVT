import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import pearsonr, mannwhitneyu, shapiro, ttest_ind

def describe_df (data: pd.DataFrame) -> pd.DataFrame: 
    '''
    Describe el dtype de cada columna, los valores nulos en %, quantos valores únicos hay en la columna y el % de cardinalidad.

    Argumentos:
    data(pd.DataFrame): DataFrame de Pandas inicial

    Retorna:
    pd.DataFrame: Data inicial transformado con los valores descritos   
    '''

    dic_describe = {
        'DATA_TYPE' : [data[x].dtype for x in data],
        'MISSINGS (%)' : [round(data[x].isnull().sum()/len(data[x])*100,2) for x in data],
        'UNIQUE_VALUES' : [data[x].nunique() for x in data],
        'CARDIN (%)' : [round(data[x].nunique() / len(data[x]) * 100, 2) for x in data]
    }
    
    return pd.DataFrame(dic_describe, index=[x for x in data]).T

###############################################################################################

def _classify(data: pd.DataFrame, key: str,  umbral_categoria:int, umbral_continua:float) -> str: 
    cardi = data[key].nunique() 
    if cardi == 2: 
        return "Binaria"
    elif cardi < umbral_categoria: 
        return "Categórica"
    elif cardi/len(data[key])*100 >= umbral_continua: 
        return "Numérica Continua"
    else:
        return "Numérica Discreta"
        
def tipifica_variables (data:pd.DataFrame, umbral_categoria:int, umbral_continua:float) -> pd.DataFrame:
    '''
    Tipo de variable de cada columna según su cardinalidad.

    Argumentos:
    data(pd.DataFrame): DataFrame inicial
    umbral_categoria(int): Número que escogemos para delimitar a partir de cuanto consideramos que es una variable categorica
    umbral_continua(float): Número que escogemos para delimitar a partir de cuanto una variable numérica es discreta
    
    Retorna:
    pd.DataFrame: Data inicial transformado   
    '''

    dic_tip_var = {
        "tipo_sugerido": [_classify(data, key, umbral_categoria, umbral_continua) for key in data]
    }
    for x in data:
        hay_nulos = data[x].isnull().sum()
        if hay_nulos != 0:
            print(f'OJO! En la columna "{x}" hay valores nulos.')

    return pd.DataFrame(dic_tip_var, index=[x for x in data])

###############################################################################################

def get_features_num_regression (df:pd.DataFrame, target_col:str, umbral_corr:float, pvalue = None):

    """
    Está función debe devolver una lista con las columnas numéricas del dataframe cuya correlación con la columna designada por "target_col" 
    sea superior en valor absoluto al valor dado por "umbral_corr". Además si la variable "pvalue" es distinta de None, sólo devolverá las 
    columnas numéricas cuya correlación supere el valor indicado y además supere el test de hipótesis con significación mayor o igual a 1-pvalue.

    Argumentos:
    - df (DataFrame): un dataframe de Pandas.
    - target_col (string): el nombre de la columna del Dataframe objetivo.
    - umbral_corr (float): un valor de correlación arbitrario sobre el que se elegirá como de correlacionadas queremos que estén las columnas elegidas (por defecto 0).
    - pvalue (float): con valor "None" por defecto.

    Retorna:
    - Lista de las columnas correlacionadas que cumplen el test en caso de que se haya pasado p-value.
    """

    # Comprobaciones

    # Comprobar si target_col está en el DataFrame
    if target_col not in df.columns:
        print(f"Error: La columna {target_col} no está en el DataFrame.")
        return None
    
    # Comprobar si target_col es numérico
    if not np.issubdtype(df[target_col].dtype, np.number):
        print(f"Error: La columna {target_col} no es numérica.")
        return None
    
    # Comprobar si umbral_corr está entre 0 y 1
    if not 0 <= umbral_corr <= 1:
        print("Error: El umbral_corr debe estar entre 0 y 1.")
        return None
    
    #Comprobar que p-value es un entero o un float, y esta en el rango [0,1]
    if pvalue is not None:
        if type(pvalue) != float and type(pvalue) != int:
            print("Error: El parámetro pvalue", pvalue, " no es un número.")
            return None
        elif  not (0 <= pvalue <= 1):
            print("Error: El parametro pvalue", pvalue, " está fuera del rango [0,1].")
            return None
        
    # Se usa la función tipifica_variables para identificar las variables numéricas
    var_tip = tipifica_variables(df, 5, 9)
    col_num = var_tip[(var_tip["tipo_sugerido"] == "Numérica Continua") | (var_tip["tipo_sugerido"] == "Numérica Discreta")]["nombre_variable"].tolist()

    # Comprobación de que hay alguna columna numérica para relacionar
    if len(col_num) == 0:
        print("Error: No hay ninguna columna númerica o discreta a analizar que cumpla con los requisitos establecidos en los umbrales.")
    else:

    # Se realizan las correlaciones y se eligen las que superen el umbral
        correlaciones = df[col_num].corr()[target_col]
        columnas_filtradas = correlaciones[abs(correlaciones) > umbral_corr].index.tolist()
        if target_col in columnas_filtradas:
            columnas_filtradas.remove(target_col)
    
        # Comprobación de que si se introduce un p-value pase los tests de hipótesis (Pearson)
        if pvalue is not None:
            columnas_finales = []
            for col in columnas_filtradas:
                p_value_especifico = pearsonr(df[col], df[target_col])[1]
                if pvalue < (1 - p_value_especifico):
                    columnas_finales.append(col)
            columnas_filtradas = columnas_finales.copy()

    if len(columnas_filtradas) == 0:
        print("No hay columna numérica que cumpla con las especificaciones de umbral de correlación y/o p-value.")
        return None

    return columnas_filtradas

###############################################################################################

def plot_features_num_regression(df:pd.DataFrame, target_col = "", columns = [], umbral_corr = 0, pvalue = None):
    
    """
    Está función pintará una pairplot del dataframe considerando la columna designada por "target_col" y aquellas 
    incluidas en "column" que cumplan que su correlación con "target_col" es superior en valor absoluto a "umbral_corr", 
    y que, en el caso de ser pvalue diferente de "None", además cumplan el test de correlación para el nivel 1-pvalue de significación estadística. 
    La función devolverá los valores de "columns" que cumplan con las condiciones anteriores.

    Argumentos:
    - df (DataFrame): un dataframe de Pandas.
    - target_col (string): el nombre de la columna del Dataframe objetivo.
    - columns (list): una lista de strings cuyo valor por defecto es la lista vacía.
    - umbral_corr (float): un valor de correlación arbitrario sobre el que se elegirá como de correlacionadas queremos que estén las columnas elegidas (por defecto 0).
    - pvalue (float): con valor "None" por defecto.

    Retorna:
    - Pairplots: columnas correlacionadas y la columna objetivo bajo nuestro criterio.
    - Lista de las columnas correlacionadas.
    """

    # Comprobaciones
    
    # Si la lista de columnas está vacía, asignar todas las variables numéricas del dataframe
    if not columns:
        columns = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
    
    # Si se proporciona una lista de columnas, comprobar si están en el DataFrame
    else:
        for col in columns:
            if col not in df.columns:
                print(f"Error: La columna {col} no está en el DataFrame.")
                return None
    
    columnas_filtradas = get_features_num_regression(df, target_col, umbral_corr, pvalue)
    
    columnas_refiltradas = []
    for col in columnas_filtradas:
        for col2 in columns:
            if col == col2:
                columnas_refiltradas.append(col)

    # Divide la lista de columnas filtradas en grupos de máximo cinco columnas
    columnas_agrupadas = [columnas_refiltradas[i:i+4] for i in range(0, len(columnas_refiltradas), 4)]
    
    # Generar pairplots para cada grupo de columnas
    for group in columnas_agrupadas:
        sns.pairplot(df[[target_col] + group])
        plt.show()
    
    # Devolver la lista de columnas filtradas
    return columnas_refiltradas

###############################################################################################

def get_features_cat_regression(df:pd.DataFrame, target_col:str, pvalue = 0.05):
    
    #Comprobar que p-value es un entero o un float, y esta en el rango [0,1]
    if type(pvalue) != float and type(pvalue) != int:
        print("Error: El parámetro pvalue", pvalue, " no es un número.")
        return None
    elif  not (0 <= pvalue <= 1):
        print("Error: El parametro pvalue", pvalue, " está fuera del rango [0,1].")
        return None
        
    #Comprobar que target_col es una variable del dataframe
    if  not (target_col in df.columns):
        print("Error: El parámetro target ", target_col , " no es una columna del Dataframe.")
        return None  
      
    #Comprobar que target_col es una variable numérica
    var_tip = tipifica_variables(df, 5, 9)

    if not (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Continua") or (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Discreta"):
        print("Error: El parámetro target_col ", target_col ,"no es una columna numérica del dataframe bajo los criterios de umbrales establecidos.")
        return None

    #Hacer una lista con las colmunnas categóricas o binarias
    col_cat = var_tip[(var_tip["tipo_sugerido"] == "Categórica") | (var_tip["tipo_sugerido"] == "Binaria")]["nombre_variable"].tolist()
    if col_cat == 0:
        return None
         
    #Inicializamos la lista de salida
    col_selec = []
    
    #Por cada columna categórica o binaria
    for valor in col_cat:
        grupos = df[valor].unique()  # Obtener los valores únicos de la columna categórica
        if len(grupos) == 2:
            grupo_a = df.loc[df[valor] == grupos[0]][target_col]
            grupo_b = df.loc[df[valor] == grupos[1]][target_col]
            u_stat, p_val = mannwhitneyu(grupo_a, grupo_b)  # Aplicamos el test U de Mann
        else:
            v_cat = [df[df[valor] == grupo][target_col] for grupo in grupos] # obtenemos los grupos y los incluimos en una lista
            f_val, p_val = stats.f_oneway(*v_cat) # Aplicamos el test ANOVA. El método * (igual que cuando vimos *args hace mil años)
        if p_val < pvalue:
            col_selec.append(valor) #Si supera el test correspondiente añadimos la variable a la lista de salida

    if len(col_selec) == 0:
        print("No hay columna categórica o binaria que cumpla con las especificaciones.")
        return None
       
    return col_selec

###############################################################################################

def plot_features_cat_regression(df:pd.DataFrame, target_col= "", columns=[], pvalue=0.05):
    
    # Comprobar que p-value es un entero o un float, y esta en el rango [0,1]
    if type(pvalue) != float and type(pvalue) != int:
        print("Error: El parámetro pvalue", pvalue, "no es un número.")
        return None
    elif not (0 <= pvalue <= 1):
        print("Error: El parámetro pvalue", pvalue, "está fuera del rango [0,1].")
        return None
      
    # Comprobar que target_col es una variable numérica continua
    var_tip = tipifica_variables(df, 5, 9)

    # Si no hay target_col, pedir al usuario la introducción de una
    if target_col == "":
        print("Por favor, introduce una columna objetivo con la que realizar el análisis.")
        return "plot_features_cat_regression(df, target_col= ___, ...)"

    # Comprobar que target_col es una variable del dataframe
    if  not (target_col in df.columns):
        print("Error: El parámetro target ", target_col , " no es una columna del Dataframe.")
        return None  

    if not (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Continua") or (var_tip.loc[var_tip["nombre_variable"] == target_col, "tipo_sugerido"].iloc[0] == "Numérica Discreta"):
        print("Error: El parámetro target_col ", target_col , " no es una columna numérica del dataframe bajo los criterios de umbrales establecidos.")
        return None
    

    # Si la lista de columnas está vacía, asignar todas las variables CATEGORICAS del dataframe
    if not columns:
        columns = var_tip[var_tip["tipo_sugerido"] == "Categórica"]["nombre_variable"].tolist()
    
    # Si se proporciona una lista de columnas, comprobar si están en el DataFrame
    else:
        for col in columns:
            if col not in df.columns:
                print(f"Error: La columna {col} no está en el DataFrame.")
                return None    

    df_columns = df[columns]
    df_columns[target_col] = df[target_col]        
    
    columnas_filtradas = get_features_cat_regression(df_columns, target_col, pvalue)

    # Generar los histogramas agrupados para cada columna filtrada
    for col in columnas_filtradas:        
        fig = sns.histplot(data=df, x=col, hue=target_col, multiple='stack')
        plt.title(f'Histograma agrupado para {col} en relación con {target_col}')
        plt.xlabel(col)
        plt.show()
    
    # Devolver la lista de columnas filtradas
    return columnas_filtradas