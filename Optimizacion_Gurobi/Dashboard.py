import streamlit as st 
import numpy as np 
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

st.title("Gestión de Operaciones")

st.write("""
# Problema de Producción y Mezcla de Café
""")

# Importacion de Datos
# Importamos los tipos de Cafe
dfTiposCafe = pd.read_excel('Data.xlsx', index_col = 0)
st.write("""
## Tipos de Cafe
""")
st.table(dfTiposCafe)
# Importamos los tipos de Cafe
dfMezclas = pd.read_excel('Data.xlsx', index_col = 0, sheet_name = 'Mezclas')
st.write("""
## Tipos de Mezcla
""")
st.table(dfMezclas)

# Conjuntos
st.write("""
## Conjuntos
""")
# Tipos de Cafe
tiposCafe = list(dfTiposCafe.index)
st.write('Tipos de Cafe:', ', '.join(tiposCafe))
# Tipos de Mezcla
mezclas = list(dfMezclas.index)
st.write('Tipos de Mezcla:', ', '.join(mezclas))

# Variables de Decision
st.write("""
## Variables de Decision
""")
st.image('variables-cafe.gif', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
arcos = [(i,j) for i in tiposCafe for j in mezclas]
for i, j in arcos:
    st.write('Cantidad de libras del pais', i, 'para la mezcla', j)

# Funcion Objetivo
st.write("""
## Función Objetivo
""")
st.image('funcion_objetivo.gif', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# Restricciones
st.write("""
## Restricciones
""")
st.write("""
### Disponibilidad de Cafe
""")
st.image('disponibilidad-cafe.gif', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.write("""
### Demanda de Mezclas
""")
st.image('demanda-mezcla-cafe.gif', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
st.write("""
### Porcentaje Máximo de Cafeina
""")
st.image('porcentaje-maximo-cafeina.gif', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# Datos del problema
costoxlibra = dfTiposCafe['Costo_Libra'].to_dict()
pctcafeina = dfTiposCafe['PCT_Cafeina'].to_dict()
librasdisponibles = dfTiposCafe['Libras_Disponibles'].to_dict()
precioventa = dfMezclas['Precio_Venta'].to_dict()
pctmaxcafeina = dfMezclas['PCT_Max_Cafeina'].to_dict()
demanda = dfMezclas['Demanda'].to_dict()

# Modelo
m = gp.Model('Factory Planning I')
# Variables de Decision
x = m.addVars(arcos, vtype = GRB.CONTINUOUS, name = 'x') # quantity manufactured
# Funcion Objetivo
m.setObjective(gp.quicksum([x[(i, j)] * ( precioventa[j] - costoxlibra[i] ) for i in tiposCafe for j in mezclas]), GRB.MAXIMIZE)
# Restricciones
m.addConstrs(gp.quicksum(x[(i,j)] for j in mezclas) <= librasdisponibles[i] for i in tiposCafe) # Disponibilidad de Cafe
m.addConstrs(gp.quicksum(x[(i,j)] for i in tiposCafe) == demanda[j] for j in mezclas) # Demanda de Mezcla
m.addConstrs(gp.quicksum(x[(i,j)] * pctcafeina[i] for i in tiposCafe) <= pctmaxcafeina[j] * gp.quicksum(x[(i,j)] for i in tiposCafe) for j in mezclas) # Porcentaje de Cafeina
# Optimizando...
m.optimize()

st.write("""
# Optimización del Modelo
""")

col1, col2, col3 = st.columns(3)
col2.metric(label="Funcion Objetivo", value = m.ObjVal, delta=None)

for v in m.getVars():
    st.write(v.VarName, '=', v.x)