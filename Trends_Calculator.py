# -*- coding: utf-8 -*-
"""

################################ TRENDS CALCULATOR ############################

"""

import numpy as np
from scipy import stats

def MHW_trend(array):
    # Obtener las dimensiones del array
    number_years = array.shape[2]
    
    # Crear arrays para almacenar las tendencias decadales y la significancia
    decadal_trends = np.zeros((array.shape[0], array.shape[1]))
    significance = np.zeros((array.shape[0], array.shape[1]))
    
    # Calcular la tendencia decadal y la significancia para cada píxel
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # Obtener los valores de la serie temporal para el píxel actual
            temporal_serie = array[i, j, :]
            
            # Calcular los años correspondientes a cada punto de datos
            years = np.arange(0, number_years)
            
            # Encontrar índices de valores no NaN en la serie temporal
            indices_no_nan = np.isfinite(temporal_serie)
            
            if np.sum(indices_no_nan) > 1:
                # Si hay al menos 2 puntos no NaN, calcular la regresión lineal
                slope, intercept, r_value, p_value, std_err = stats.linregress(years[indices_no_nan], temporal_serie[indices_no_nan])
                
                # Calcular la tendencia decadal
                decadal_trend = slope * 10
                
                # Calcular la significancia
                if p_value >= 0.05:
                    significance_value = np.nan
                else:
                    significance_value = 1
            else:
                # Si no hay suficientes puntos no NaN, establecer la tendencia decadal y la significancia como NaN
                decadal_trend = np.nan
                significance_value = np.nan
            
            # Almacenar la tendencia decadal y la significancia en los arrays de resultados
            decadal_trends[i, j] = decadal_trend
            significance[i, j] = significance_value
    
    return decadal_trends, significance



# Calcular la tendencia decadal y la significancia

Freq_trend_AL, significance_AL = MHW_trend(MHW_cnt_ts_AL_SAT)






