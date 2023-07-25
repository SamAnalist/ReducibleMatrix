import math

import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as t
import seaborn as sns
import pandas as pd
import sqlite3
import openpyxl
import statistics
from sklearn.linear_model import LinearRegression



def correlacion(a1, a2):
    prod = np.dot(a1, a2)
    n = len(a1)
    x = np.sum(a1)
    y = np.sum(a2)
    x2 = np.sum(np.power(a1,2))
    y2 = np.sum(np.power(a2,2))
    r = (n*prod - x*y)/ ((n*x2 - x**2)*(n*y2 - y**2))**0.5
    return r
# Lee el archivo XLSX
df = pd.read_excel('ref.xlsx', sheet_name='Hoja1')

# Imprime la columna como un array de tipo float
"""
correlation_matrix = nuevo_df[['EDAD', 'PESO']].corr()
# Crear el gráfico de correlación utilizando seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# Ajustar las etiquetas de los ejes
plt.xlabel('EDAD')
plt.ylabel('PESO')

plt.scatter(nuevo_df['EDAD'], nuevo_df['PESO'])

# Ajustar las etiquetas de los ejes
plt.xlabel('EDAD')
plt.ylabel('PESO')
# Mostrar el gráfico
plt.show()
"""
"""
xs = np.array(arr2, dtype= float)
ys = np.array(arr1, dtype= float)
capas = t.keras.layers.Dense(units = 1, input_shape=[1])
modelo = t.keras.Sequential([capas])
modelo.compile(
    optimizer=t.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)
print("Comenzando entrenamiento...")
historial = modelo.fit(xs, ys, epochs=1000, verbose=False)
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
res = modelo.predict([14])
print(res)
print(capas.get_weights())
"""
########
"""
slope, intercept = np.polyfit(arr2, arr1, 1)

# Crear el gráfico de dispersión (scatter plot) de los datos
plt.scatter(arr2, arr1, color='b', label='Datos')

# Calcular los valores predichos de y utilizando los coeficientes de la regresión
y_pred = slope * arr2 + intercept
print(slope, intercept)
# Trazar la línea de regresión
plt.plot(arr2, y_pred, color='r', label='Regresión lineal')

# Ajustar las etiquetas de los ejes y el título del gráfico
plt.xlabel('EDAD')
plt.ylabel('PESO')
plt.title('Regresión Lineal')

# Mostrar la leyenda del gráfico
plt.legend()

# Mostrar el gráfico
plt.show()

# Supongamos que ya tienes un DataFrame llamado df con una columna 'edad'

# Contar el número de personas para cada edad
conteo_edades = muestra.groupby('EDAD').size()

# Imprimir el conteo de edades
print(conteo_edades)
"""
df['FECHA'] = pd.to_datetime(df['FECHA'])
# Definir la condición para la muestra (por ejemplo, fechas entre '2022-01-01' y '2022-12-31')
fecha_inicio = pd.to_datetime('2019/01/01')
fecha_fin = pd.to_datetime('2023/12/31')
condicion = (df['FECHA'] >= fecha_inicio) & (df['FECHA'] <= fecha_fin)  & (df['VIVI/MUERTO'] != 'MUERTO')
# Obtener una muestra que cumpla con la condición
muestra2 = df[condicion].sample(n=1080, random_state=60).sort_values('FECHA')
muestra2.to_excel('MuestraPorFecha.xlsx', index=False)
conteo_edades = muestra2.groupby('EDAD').size()
personas_menores_20 = muestra2[muestra2['EDAD'] <= 20]

# Contar cuántas personas hay menores de 20 años
conteo_personas_menores_20 = len(personas_menores_20)
print(conteo_personas_menores_20)
# Imprimir el conteo de edades
#print(conteo_edades)
# Imprimir la muestra

media = muestra2['EDAD'].mean()

# Calcular la mediana
mediana = muestra2['EDAD'].median()

# Calcular la moda
moda = muestra2['EDAD'].mode()

# Imprimir los resultados


# Definir los límites de las clases con una amplitud de 3 años
limites_clases = range(12, 50, 7)  # Intervalos de 3 en 3 años

# Crear una nueva columna 'clase' en el DataFrame con los intervalos de edades
muestra2['clase'] = pd.cut(muestra2['EDAD'], bins=limites_clases, right=False)

# Calcular las frecuencias simples
frecuencia_simple = muestra2['clase'].value_counts().sort_index()

# Calcular las frecuencias relativas
frecuencia_relativa = frecuencia_simple / len(muestra2)

# Calcular las frecuencias acumuladas
frecuencia_acumulada = frecuencia_simple.cumsum()

# Crear una nueva tabla con los datos agrupados
dat_agrupados = pd.DataFrame({
    'Clase': frecuencia_simple.index,
    'Frecuencia Simple': frecuencia_simple.values,
    'Frecuencia Relativa': frecuencia_relativa.values,
    'Frecuencia Acumulada': frecuencia_acumulada.values
})

# Imprimir la nueva tabla
dat_agrupados.to_excel('datosAgrupados.xlsx', index=False)
correlation_matrix = muestra2[['EDAD', 'EDAD GESTACIONAL', 'PESO']].corr()
print("/////////////")
# Imprimir la matriz de correlación
#print(correlation_matrix)

# Realizar una regresión múltiple para predecir 'y' utilizando 'x1', 'x2' y 'x3'
from sklearn.linear_model import LinearRegression

u = muestra2[['EDAD', 'EDAD GESTACIONAL']]
v = muestra2['PESO']

model = LinearRegression()
model.fit(u, v)

# Obtener los coeficientes de la regresión
coefficients = pd.DataFrame({'Variable': ['EDAD', 'EDAD GESTACIONAL'], 'Coefficient': model.coef_})
intercept = model.intercept_

#("f(x, y) = ", coefficients)
#print(intercept)
#print("%%%")

# Imprimir la nueva tabla
dat_agrupados.to_excel('datosAgrupados.xlsx', index=False)
correlation_matrix= muestra2[['EDAD', 'PESO']].corr()
"""
def BiaMar(edad, edad_gestacional):
    r = 0.005314*edad + 0.068769*edad_gestacional +0.3783777703234801
    return r

# Imprimir la matriz de correlación

#

#print(correlation_matrix)


# Realizar una regresión múltiple para predecir 'y' utilizando 'x1', 'x2' y 'x3'
from sklearn.linear_model import LinearRegression

u = muestra2[['EDAD GESTACIONAL']]
v = muestra2['PESO']

model = LinearRegression()
model.fit(u, v)

# Obtener los coeficientes de la regresión
co= pd.DataFrame({'Variable': ['EDAD GESTACIONAL'], 'Coefficient': model.coef_})
int = model.intercept_
#("Coeficiente edad gestacional: ",co)
#print("b ", int)

from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

######################################## Data preparation #########################################


X = df[['EDAD', 'EDAD GESTACIONAL']].values.reshape(-1,2)
Y = df['PESO']

######################## Prepare model data point for visualization ###############################

x = X[:, 0]
y = X[:, 1]
z = Y

x_pred = np.linspace(6, 24, 30)   # range of porosity values
y_pred = np.linspace(0, 100, 30)  # range of brittleness values
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

################################################ Train #############################################

ols = linear_model.LinearRegression()
model = ols.fit(X, Y)
predicted = model.predict(model_viz)

############################################## Evaluate ############################################

r2 = model.score(X, Y)

############################################## Plot ################################################

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('EDAD ', fontsize=12)
    ax.set_ylabel('EDAD GESTACIONAL', fontsize=12)
    ax.set_zlabel('PESO', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')

ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)

fig.tight_layout()
plt.show()




# Supongamos que ya tienes un DataFrame llamado df con las variables 'x' y 'y'
######################################## Data preparation #########################################


X = muestra2['EDAD'].values.reshape(-1,1)
y = muestra2['PESO'].values

################################################ Train #############################################

ols = linear_model.LinearRegression()
model = ols.fit(X, y)
response = model.predict(X)

############################################## Evaluate ############################################

r2 = model.score(X, y)

############################################## Plot ################################################

plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(X, response, color='k', label='Regresión lineal')
ax.scatter(X, y, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
ax.set_ylabel('PESO', fontsize=14)
ax.set_xlabel('EDAD', fontsize=14)
ax.legend(facecolor='white', fontsize=11)
ax.set_title('Modelo de regresión lineal' % r2, fontsize=18)

fig.tight_layout()
plt.show()
"""
def BiaMar1(edad):
    r = 0.00599*edad + 2.988655026006851
    return r

#print(BiaMar1(15))
#print(BiaMar(45, 37))
"""
datos_menores = muestra2[df['EDAD'] < 20]

# Crear una nueva tabla con los datos filtrados
muestra_BP = datos_menores.copy()
# Guardar la nueva tabla en un archivo
muestra_BP.to_csv('MuestraMenores20.csv', index=False)

datos_menores = datos_menores[df['PESO'] <= 2.5]

# Crear una nueva tabla con los datos filtrados
muestra_BP = datos_menores.copy()
# Guardar la nueva tabla en un archivo
muestra_BP.to_csv('MuestraBajoPeso1.csv', index=False)
conteo_edades = muestra2.groupby('EDAD').size()
conto2 = muestra2.groupby('EDAD GESTACIONAL').size()
#print("Frecuencia edad gestacional: ",conto2)
#print("Frecuencia edad de la madre: ", conteo_edades)

# Imprimir el conteo de edades
def BiaMar2(edadGestacional):
    r = 0.0691*edadGestacional +0.4919081052562282
    return r

import pandas as pd
from scipy.stats import shapiro

# Supongamos que ya tienes un DataFrame llamado df con una columna 'datos'
"""
# Extraer los datos a analizar
"""
datos = df['PESO']

# Realizar la prueba de Shapiro-Wilk
stat, p_value = shapiro(datos)

# Imprimir los resultados
print("Estadística de prueba:", stat)
print("Valor p:", p_value)

# Realizar una interpretación de los resultados
alpha = 0.05  # Nivel de significancia
if p_value > alpha:
    print("Los datos parecen seguir una distribución normal")
else:
    print("Los datos no parecen seguir una distribución normal")

muestra1 = df[df['EDAD GESTACIONAL'] >= 40].sample(frac=1, replace=False)
muestra1.to_excel('MuestraEdadgestMayor40.xlsx', index=False)
fecha_inicio = pd.to_datetime('2022/01/01')
fecha_fin = pd.to_datetime('2022/12/31')
condicion = (muestra2['FECHA'] >= fecha_inicio) & (muestra2['FECHA'] <= fecha_fin)  & (muestra2['VIVI/MUERTO'] != 'MUERTO')
# Obtener una muestra que cumpla con la condición
muestra2019 = muestra2[condicion].sample(frac=1, replace=False).sort_values('FECHA')
muestra2019.to_excel('Muestra2022.xlsx', index=False)
limites_clases = range(11, 53, 4)  # Intervalos de 3 en 3 años
muestra2019['frecuencia_bajo_peso'] = (muestra2019['PESO'] <= 2.5)
# Crear una nueva columna 'clase' en el DataFrame con los intervalos de edades
muestra2019['clase'] = pd.cut(muestra2019['EDAD'], bins=limites_clases, right=False)

# Calcular las frecuencias simples
frecuencia_simple = muestra2019['clase'].value_counts().sort_index()
# Calcular las frecuencias relativas
frecuencia_relativa = frecuencia_simple / len(muestra2019)
# Filtrar los datos con peso menor a 65 kg
muestra_filtrada = muestra2019[muestra2019['PESO'] <=2.5]

# Calcular las frecuencias simples de pesos menores a 65 en cada intervalo de edad
frecuencia_simple_peso_menor_65 = muestra_filtrada['clase'].value_counts().sort_index()
total_peso_menor_65 = len(muestra_filtrada)
frecuencia_relativa_peso_menor_65 = frecuencia_simple_peso_menor_65 / total_peso_menor_65

# Añadir la columna de frecuencia simple de pesos menores a 65 a la tabla de datos agrupados

# Calcular las frecuencias acumuladas
frecuencia_acumulada = frecuencia_simple.cumsum()

# Crear una nueva tabla con los datos agrupados
dat_agrupados = pd.DataFrame({
    'Clase': frecuencia_simple.index,
    'Frecuencia (EG)': frecuencia_simple.values,
    'Frecuencia Relativa': frecuencia_relativa.values,
    'Frecuencia Acumulada': frecuencia_acumulada.values,
    'Frecuencia (BP)': frecuencia_simple_peso_menor_65.values,
    'Frecuencia Relativa (BP)': frecuencia_relativa_peso_menor_65
})

# Imprimir la nueva tabla
dat_agrupados.to_excel('datosAgrupadosEDAD2022.xlsx', index=False)


fecha_inicio = pd.to_datetime('2021/01/01')
fecha_fin = pd.to_datetime('2021/12/31')
condicion = (muestra2['FECHA'] >= fecha_inicio) & (muestra2['FECHA'] <= fecha_fin)  & (muestra2['VIVI/MUERTO'] != 'MUERTO')
# Obtener una muestra que cumpla con la condición
muestra2019 = muestra2[condicion].sample(frac=1, replace=False).sort_values('FECHA')
muestra2019.to_excel('Muestra2021.xlsx', index=False)
limites_clases = range(11, 53, 4)  # Intervalos de 3 en 3 años
muestra2019['frecuencia_bajo_peso'] = (muestra2019['PESO'] <= 2.5)
# Crear una nueva columna 'clase' en el DataFrame con los intervalos de edades
muestra2019['clase'] = pd.cut(muestra2019['EDAD'], bins=limites_clases, right=False)

# Calcular las frecuencias simples
frecuencia_simple = muestra2019['clase'].value_counts().sort_index()
# Calcular las frecuencias relativas
frecuencia_relativa = frecuencia_simple / len(muestra2019)
# Filtrar los datos con peso menor a 65 kg
muestra_filtrada = muestra2019[muestra2019['PESO'] <=2.5]

# Calcular las frecuencias simples de pesos menores a 65 en cada intervalo de edad
frecuencia_simple_peso_menor_65 = muestra_filtrada['clase'].value_counts().sort_index()
total_peso_menor_65 = len(muestra_filtrada)
frecuencia_relativa_peso_menor_65 = frecuencia_simple_peso_menor_65 / total_peso_menor_65

# Añadir la columna de frecuencia simple de pesos menores a 65 a la tabla de datos agrupados

# Calcular las frecuencias acumuladas
frecuencia_acumulada = frecuencia_simple.cumsum()

# Crear una nueva tabla con los datos agrupados
dat_agrupados = pd.DataFrame({
    'Clase': frecuencia_simple.index,
    'Frecuencia (EG)': frecuencia_simple.values,
    'Frecuencia Relativa': frecuencia_relativa.values,
    'Frecuencia Acumulada': frecuencia_acumulada.values,
    'Frecuencia (BP)': frecuencia_simple_peso_menor_65.values,
    'Frecuencia Relativa (BP)': frecuencia_relativa_peso_menor_65
})

# Imprimir la nueva tabla
dat_agrupados.to_excel('datosAgrupadosEDAD2021.xlsx', index=False)



##
fecha_inicio = pd.to_datetime('2020/01/01')
fecha_fin = pd.to_datetime('2020/12/31')
condicion = (muestra2['FECHA'] >= fecha_inicio) & (muestra2['FECHA'] <= fecha_fin)  & (muestra2['VIVI/MUERTO'] != 'MUERTO')
# Obtener una muestra que cumpla con la condición
muestra2019 = muestra2[condicion].sample(frac=1, replace=False).sort_values('FECHA')
muestra2019.to_excel('Muestra2020.xlsx', index=False)
limites_clases = range(11, 53, 4)  # Intervalos de 3 en 3 años
muestra2019['frecuencia_bajo_peso'] = (muestra2019['PESO'] <= 2.5)
# Crear una nueva columna 'clase' en el DataFrame con los intervalos de edades
muestra2019['clase'] = pd.cut(muestra2019['EDAD'], bins=limites_clases, right=False)

# Calcular las frecuencias simples
frecuencia_simple = muestra2019['clase'].value_counts().sort_index()
# Calcular las frecuencias relativas
frecuencia_relativa = frecuencia_simple / len(muestra2019)
# Filtrar los datos con peso menor a 65 kg
muestra_filtrada = muestra2019[muestra2019['PESO'] <=2.5]

# Calcular las frecuencias simples de pesos menores a 65 en cada intervalo de edad
frecuencia_simple_peso_menor_65 = muestra_filtrada['clase'].value_counts().sort_index()
total_peso_menor_65 = len(muestra_filtrada)
frecuencia_relativa_peso_menor_65 = frecuencia_simple_peso_menor_65 / total_peso_menor_65

# Añadir la columna de frecuencia simple de pesos menores a 65 a la tabla de datos agrupados

# Calcular las frecuencias acumuladas
frecuencia_acumulada = frecuencia_simple.cumsum()

# Crear una nueva tabla con los datos agrupados
dat_agrupados = pd.DataFrame({
    'Clase': frecuencia_simple.index,
    'Frecuencia (EG)': frecuencia_simple.values,
    'Frecuencia Relativa': frecuencia_relativa.values,
    'Frecuencia Acumulada': frecuencia_acumulada.values,
    'Frecuencia (BP)': frecuencia_simple_peso_menor_65.values,
    'Frecuencia Relativa (BP)': frecuencia_relativa_peso_menor_65
})

# Imprimir la nueva tabla
dat_agrupados.to_excel('datosAgrupadosEDAD2020.xlsx', index=False)

##
fecha_inicio = pd.to_datetime('2019/01/01')
fecha_fin = pd.to_datetime('2019/12/31')
condicion = (muestra2['FECHA'] >= fecha_inicio) & (muestra2['FECHA'] <= fecha_fin)  & (muestra2['VIVI/MUERTO'] != 'MUERTO')
# Obtener una muestra que cumpla con la condición
muestra2019 = muestra2[condicion].sample(frac=1, replace=False).sort_values('FECHA')
muestra2019.to_excel('Muestra2019.xlsx', index=False)
limites_clases = range(11, 53,4 )  # Intervalos de 3 en 3 años
muestra2019['frecuencia_bajo_peso'] = (muestra2019['PESO'] <= 2.5)
# Crear una nueva columna 'clase' en el DataFrame con los intervalos de edades
muestra2019['clase'] = pd.cut(muestra2019['EDAD'], bins=limites_clases, right=False)

# Calcular las frecuencias simples
frecuencia_simple = muestra2019['clase'].value_counts().sort_index()
# Calcular las frecuencias relativas
frecuencia_relativa = frecuencia_simple / len(muestra2019)
# Filtrar los datos con peso menor a 65 kg
muestra_filtrada = muestra2019[muestra2019['PESO'] <=2.5]

# Calcular las frecuencias simples de pesos menores a 65 en cada intervalo de edad
frecuencia_simple_peso_menor_65 = muestra_filtrada['clase'].value_counts().sort_index()
total_peso_menor_65 = len(muestra_filtrada)
frecuencia_relativa_peso_menor_65 = frecuencia_simple_peso_menor_65 / total_peso_menor_65

# Añadir la columna de frecuencia simple de pesos menores a 65 a la tabla de datos agrupados

# Calcular las frecuencias acumuladas
frecuencia_acumulada = frecuencia_simple.cumsum()

# Crear una nueva tabla con los datos agrupados
dat_agrupados = pd.DataFrame({
    'Clase': frecuencia_simple.index,
    'Frecuencia (EG)': frecuencia_simple.values,
    'Frecuencia Relativa': frecuencia_relativa.values,
    'Frecuencia Acumulada': frecuencia_acumulada.values,
    'Frecuencia (BP)': frecuencia_simple_peso_menor_65.values,
    'Frecuencia Relativa (BP)': frecuencia_relativa_peso_menor_65
})

# Imprimir la nueva tabla
dat_agrupados.to_excel('datosAgrupadosEDAD2019.xlsx', index=False)


"""

datos_menores = muestra2[(muestra2['PESO'] <= 2.5)]

# Crear una nueva tabla con los datos filtrados
muestra_BP = datos_menores.copy()
# Guardar la nueva tabla en un archivo
#muestra_BP.to_csv('MuestraBajoPeso.csv', index=False)

#datos_menores = datos_menores[df['PESO'] <= 2.5]

# Crear una nueva tabla con los datos filtrados
#muestra_BP = datos_menores.copy()
# Guardar la nueva tabla en un archivo
#muestra_BP.to_csv('MuestraBajoPeso.csv', index=False)

# Convertir la columna 'FECHA_NACIMIENTO' a tipo datetime
df['FECHA'] = pd.to_datetime(df['FECHA'])
# Definir la condición para la muestra (por ejemplo, fechas entre '2022-01-01' y '2022-12-31')
fecha_inicio = pd.to_datetime('2022/01/01')
fecha_fin = pd.to_datetime('2022/12/31')

# Filtrar las personas menores de 19 años en el año 2019
personas_menores_19_2019 = muestra2[(muestra2['EDAD'] <= 19) & (muestra2['FECHA'] >= fecha_inicio) & (muestra2['FECHA'] <= fecha_fin)]

# Obtener el conteo de personas menores de 19 años en el año 2019
conteo_personas_menores_19_2019 = len(personas_menores_19_2019)
print(conteo_personas_menores_19_2019)











