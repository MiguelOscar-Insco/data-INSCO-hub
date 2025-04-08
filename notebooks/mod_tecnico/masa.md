## Análisis del laboratorio de masa


```python
# Import necessary libraries
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
```


```python
# Import custom libraries
import warnings
import sys
import os

# Subir dos niveles desde notebooks/mod_tecnico/ hasta la raíz del repo
repo_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))  
sys.path.append(repo_path)

# Ahora intenta importar
from src.data_processing import ready_to_import

```


```python
#Importar librerias para graficar
from matplotlib import pyplot as plt
import seaborn as sns
```


```python
import sys
!{sys.executable} -m pip install pandasql --user
```

    "c:\Users\Miguel" no se reconoce como un comando interno o externo,
    programa o archivo por lotes ejecutable.
    

#### Importamos lo necesario para trabajar con consultas SQL


```python
from pandasql import sqldf
print("¡Listo para usar SQL en pandas!")
from pandasql import sqldf

# Definir función de consulta
pysqldf = lambda q: sqldf(q, globals())
```

    ¡Listo para usar SQL en pandas!
    


```python
# Import data
# Import data from Excel files
masa = pd.read_excel('C:/Users/Miguel Oscar/Projects/data-INSCO-hub/data/raw/masa.xlsx')
```


```python
# Resumen estadístico de la tabla
masa.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lab_received_date</th>
      <th>scheduled_delivery_date</th>
      <th>cleaning_date</th>
      <th>calibration_date</th>
      <th>delivery_date</th>
      <th>delivery_time</th>
      <th>assigned_time</th>
      <th>piece_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>461</td>
      <td>454</td>
      <td>474</td>
      <td>475</td>
      <td>480</td>
      <td>480.00</td>
      <td>480.00</td>
      <td>480.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2025-01-30 01:11:50.629067264</td>
      <td>2025-02-18 17:48:53.920704768</td>
      <td>2025-01-31 00:57:43.291139328</td>
      <td>2025-02-09 17:59:14.526315776</td>
      <td>2025-02-15 04:06:00</td>
      <td>8.65</td>
      <td>12.29</td>
      <td>5.09</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2024-11-06 00:00:00</td>
      <td>2024-12-19 00:00:00</td>
      <td>2024-03-12 00:00:00</td>
      <td>2024-03-13 00:00:00</td>
      <td>2025-01-02 00:00:00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2025-01-14 00:00:00</td>
      <td>2025-01-31 00:00:00</td>
      <td>2025-01-13 00:00:00</td>
      <td>2025-01-23 00:00:00</td>
      <td>2025-01-24 00:00:00</td>
      <td>5.00</td>
      <td>11.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2025-01-28 00:00:00</td>
      <td>2025-02-18 00:00:00</td>
      <td>2025-01-29 00:00:00</td>
      <td>2025-02-10 00:00:00</td>
      <td>2025-02-11 00:00:00</td>
      <td>8.00</td>
      <td>13.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2025-02-25 00:00:00</td>
      <td>2025-03-13 00:00:00</td>
      <td>2025-02-28 00:00:00</td>
      <td>2025-03-08 12:00:00</td>
      <td>2025-03-11 00:00:00</td>
      <td>11.00</td>
      <td>15.00</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2025-03-27 00:00:00</td>
      <td>2025-04-04 00:00:00</td>
      <td>2025-03-28 00:00:00</td>
      <td>2025-03-31 00:00:00</td>
      <td>2025-03-31 00:00:00</td>
      <td>44.00</td>
      <td>30.00</td>
      <td>58.00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.93</td>
      <td>5.24</td>
      <td>8.70</td>
    </tr>
  </tbody>
</table>
</div>



**Se cambian las columnas que contienen fechas al tipo 'datetime'**


```python
# Seleccionamos las columnas que contienen fechas y las convertimos a tipo datetime
col_fechas = [col for col in masa.columns if 'date' in col.lower() or 'fecha' in col.lower()]

for col in col_fechas:
    masa[col] = pd.to_datetime(masa[col], errors='coerce', format='%Y/%m/%d')
```


```python
masa.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 586 entries, 0 to 585
    Data columns (total 20 columns):
     #   Column                   Non-Null Count  Dtype         
    ---  ------                   --------------  -----         
     0   report_number            461 non-null    object        
     1   so                       480 non-null    object        
     2   assignee                 586 non-null    object        
     3   calibrator               479 non-null    object        
     4   supervisor               480 non-null    object        
     5   approver                 478 non-null    object        
     6   classification           480 non-null    object        
     7   lab_received_date        461 non-null    datetime64[ns]
     8   scheduled_delivery_date  454 non-null    datetime64[ns]
     9   cleaning_date            474 non-null    datetime64[ns]
     10  calibration_date         475 non-null    datetime64[ns]
     11  delivery_date            480 non-null    datetime64[ns]
     12  delivery_time            480 non-null    float64       
     13  process_status           480 non-null    object        
     14  assigned_time            480 non-null    float64       
     15  service_location         480 non-null    object        
     16  substitution_reason      25 non-null     object        
     17  lab_observations         61 non-null     object        
     18  priority                 480 non-null    object        
     19  piece_count              480 non-null    float64       
    dtypes: datetime64[ns](5), float64(3), object(12)
    memory usage: 91.7+ KB
    


```python
# Se hace una copia de la tabla para evitar problemas de referencia
mass = masa.copy()
```

**Se eliminan las filas donde 'report_number' es NaN y se une el contenido de 'assignee' de las filas con 'report_number' NaN a la fila anterior**


```python
# Esta función es interna y creada para el proceso en el que se va a usar
mass = ready_to_import.to_prepare_db(mass)
```


```python
# Verificamos nuevamente los tipos de datos de cada columna y la cantidad de datos nulos
mass.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 461 entries, 0 to 460
    Data columns (total 20 columns):
     #   Column                   Non-Null Count  Dtype         
    ---  ------                   --------------  -----         
     0   report_number            461 non-null    object        
     1   so                       461 non-null    object        
     2   assignee                 461 non-null    object        
     3   calibrator               460 non-null    object        
     4   supervisor               461 non-null    object        
     5   approver                 459 non-null    object        
     6   classification           461 non-null    object        
     7   lab_received_date        461 non-null    datetime64[ns]
     8   scheduled_delivery_date  454 non-null    datetime64[ns]
     9   cleaning_date            455 non-null    datetime64[ns]
     10  calibration_date         456 non-null    datetime64[ns]
     11  delivery_date            461 non-null    datetime64[ns]
     12  delivery_time            461 non-null    float64       
     13  process_status           461 non-null    object        
     14  assigned_time            461 non-null    float64       
     15  service_location         461 non-null    object        
     16  substitution_reason      25 non-null     object        
     17  lab_observations         61 non-null     object        
     18  priority                 461 non-null    object        
     19  piece_count              461 non-null    float64       
    dtypes: datetime64[ns](5), float64(3), object(12)
    memory usage: 72.2+ KB
    

### Comparamos la cantidad de calibraciones con los mantenimientos


```python
# Clasificar por tipo de servicio
mass['service_type'] = np.where(
    mass['report_number'].str.startswith('RSM', na=False),
    'Mantenimiento',
    np.where(
        mass['report_number'].str.startswith('CCM', na=False),
        'Calibración',
        'Otros'  # Valor por defecto si ninguna condición se cumple
    )
)
```


```python
service_type = mass['service_type'].value_counts()
service_type
```


```python
# Crear figura
plt.figure(figsize=(10, 6))

# Gráfico de barras con Seaborn
ax = sns.barplot(x=service_type.index, y=service_type.values, hue=service_type.index, palette='Blues_r', legend=False)

# Rotar etiquetas del eje X
plt.xticks(rotation=45)

# Etiquetas de los ejes
plt.xlabel('Tipo de servicio')
plt.ylabel('Catidad')
plt.title('Clasificación por tipo de servicio', fontsize=16, fontweight='bold')

# Agregar etiquetas en las barras
for i, v in enumerate(service_type.values):
    ax.text(i, v + 0.05, str(v), ha='center', fontsize=12, fontweight='bold')

# Mostrar gráfico
plt.show()
```

#### Añadir una columna para clasificar los equipos por tipos


```python
classification_map = {
    'I': 'Balanza', 'II': 'Balanza', 'III': 'Balanza', 'IIII': 'Balanza',
    'E1': 'Pesa', 'E2': 'Pesa', 'F1': 'Pesa', 'F2': 'Pesa',
    'M1': 'Pesa', 'M2': 'Pesa', 'M3': 'Pesa',
    '1': 'Pesa', '2': 'Pesa', '3': 'Pesa', '4': 'Pesa',
    '5': 'Pesa', '6': 'Pesa',
    'M': 'NBS', 'S': 'NBS', 'S-1': 'NBS', 'P': 'NBS', 'Q': 'NBS',
    'T': 'NBS', 'F': 'NBS',
    'ONN': 'ONN'
}

# Asignar valores usando map()
mass['equipment_type'] = mass['classification'].map(classification_map).fillna('Otros')
```

#### Analicemos brevemente los mantenimientos


```python
mass_maintenance = mass[mass['report_number'].str.startswith('RSM', na=False)]
mass_maintenance
```


```python
maintenance_per_type = mass_maintenance.groupby('equipment_type')['report_number'].count().sort_values(ascending=False)
maintenance_per_type
```


```python
# Crear figura
plt.figure(figsize=(10, 6))

# Gráfico de barras con Seaborn
ax = sns.barplot(x=maintenance_per_type.index, y=maintenance_per_type.values, hue=maintenance_per_type.index, palette='Blues_r', legend=False)

# Rotar etiquetas del eje X
plt.xticks(rotation=45)

# Etiquetas de los ejes
plt.xlabel('Clasificación')
plt.ylabel('Catidad')
plt.title('Clasificación de los mantenimientos por tipo de equipo', fontsize=16, fontweight='bold')

# Agregar etiquetas en las barras
for i, v in enumerate(maintenance_per_type.values):
    ax.text(i, v + 0.05, str(v), ha='center', fontsize=12, fontweight='bold')

# Mostrar gráfico
plt.show()
```


```python
# Cantidad de mantenimientos por clasificación
maintenance_class = mass_maintenance.groupby('classification')['report_number'].count().sort_values(ascending=False)
maintenance_class
```


```python
# Crear figura
plt.figure(figsize=(10, 6))

# Gráfico de barras con Seaborn
ax = sns.barplot(x=maintenance_class.index, y=maintenance_class.values, hue=maintenance_class.index, palette='Blues_r', legend=False)

# Rotar etiquetas del eje X
plt.xticks(rotation=45)

# Etiquetas de los ejes
plt.xlabel('Clasificación')
plt.ylabel('Catidad')
plt.title('Cantidad de mantenimientos por clasificación de equipos', fontsize=16, fontweight='bold')

# Agregar etiquetas en las barras
for i, v in enumerate(maintenance_class.values):
    ax.text(i, v + 0.05, str(v), ha='center', fontsize=12, fontweight='bold')

# Mostrar gráfico
plt.show()
```


```python
maintenance_per_met = (mass_maintenance.groupby(['calibrator', 'equipment_type'])
                                               .size()
                                               .unstack(fill_value=0)
                                               #.reset_index()
                                               )
maintenance_per_met
```


```python
# Configurar el estilo
plt.style.use('ggplot')

# Crear gráfico de barras apiladas
ax = maintenance_per_met.plot(
    kind='bar', 
    stacked=True,
    figsize=(14, 7),
    colormap='tab20',
    edgecolor='black',
    linewidth=0.5
)

# Calcular totales por barra
totales = maintenance_per_met.sum(axis=1)

# Añadir etiquetas de totales ENCIMA de las barras
for i, total in enumerate(totales):
    ax.text(
        x=i,                         # Posición en el eje X
        y=total + 0.5,                # Altura: total + margen
        s=f'{int(total)}',            # Texto a mostrar
        ha='center',                  # Alineación horizontal
        va='bottom',                  # Alineación vertical
        fontsize=10,
        color='black',
        weight='bold'                 # Texto en negrita
    )

# Añadir etiquetas individuales (segmentos)
for rect in ax.patches:
    height = rect.get_height()
    if height > 0:
        ax.text(
            rect.get_x() + rect.get_width()/2, 
            rect.get_y() + height/2, 
            f'{int(height)}', 
            ha='center', 
            va='center',
            fontsize=8,
            color='white'             # Color contrastante para segmentos
        )

# Personalización adicional (manteniendo el resto del código)
plt.title('Mantenimientos por metrólogo y tipo de equipo', fontsize=16, pad=20)
plt.xlabel('Metrólogo', fontsize=12)
plt.ylabel('Cantidad de mantenimientos', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.legend(title='Tipos de Error', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.show()
```

**Realizar análisis de los tiempos de los mantenimientos**


```python
maintenance_time = (mass_maintenance.groupby('classification')[['delivery_time', 'assigned_time', 'piece_count']]
                                               .mean()
                                               #.unstack(fill_value=0)
                                               #.reset_index()
                                               )  

'''maintenance_time = maintenance_time.rename(
    columns={
        'Tiempo_Entrega': 'delivery_time' ,
        'Tiempo_Asignación':'assigned_time',
        'Cantidad_Piezas':'piece_count'
    }
)'''
maintenance_time
```


```python
df = maintenance_time.reset_index()

# Configurar figura y ejes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Ancho de las barras
bar_width = 0.4
x = range(len(df))

# Barras agrupadas
bars1 = ax1.bar(x, df['delivery_time'], width=bar_width, color='skyblue', label='Tiempo de Entrega', align='center')
bars2 = ax1.bar([i + bar_width for i in x], df['assigned_time'], width=bar_width, color='salmon', label='Tiempo Asignado', align='center')

# Segundo eje Y para la cantidad de piezas
ax2 = ax1.twinx()
points = ax2.plot([i + bar_width/2 for i in x], df['piece_count'], color='black', marker='o', label='Promedio de Piezas', linestyle='dashed')

# Etiquetas en las barras
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{bar.get_height():.1f}', ha='center', fontsize=10, color='black')
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{bar.get_height():.1f}', ha='left', fontsize=10, color='black')

# Etiquetas en los puntos de cantidad de piezas
for i, txt in enumerate(df['piece_count']):
    ax2.text(i + bar_width/2, txt + 0.1, f'{txt:.1f}', ha='center', fontsize=10, color='black')

# Configurar etiquetas en el eje X
ax1.set_xticks([i + bar_width/2 for i in x])
ax1.set_xticklabels(df['classification'], rotation=45, ha='right', fontsize=10)

# Etiquetas y título
ax1.set_ylabel("Tiempo Promedio")
ax2.set_ylabel("Promedio de Piezas")
ax1.set_xlabel("Clasificación del Equipo")
ax1.set_title("Tiempo de entrega vs Tiempos asignado en los servicios de Mantenimiento", fontsize=16, fontweight='bold')

# Leyendas
ax1.legend(loc='upper left')
ax2.legend(loc='upper center')

plt.show()

```

# Realizar el análisis de las calibraciones

### Eliminamos todos los reportes que no son calibraciones


```python
# Se eliminan las filas que no son calibraciones
mass_cal = mass[~mass['report_number'].str.startswith('RSM', na=False)]
```


```python
mass_cal.info()
```

# Análisis Exploratorio de Datos (EDA)

## Estructura General del DataFrame
- **Filas**: 387 registros.
- **Columnas**: 22 en total.
- **Tipos de datos**:
  - `datetime64[ns]`: 5 columnas (fechas).
  - `float64`: 3 columnas (valores numéricos).
  - `object`: 14 columnas (texto/categorías).

---

## Valores Faltantes por Columna
| Columna                   | No. Faltantes | % Faltantes | Observación                           |
|---------------------------|---------------|-------------|---------------------------------------|
| `approver`                | 2             | 0.52%       | Datos casi completos                 |
| `scheduled_delivery_date` | 5             | 1.29%       |                                      |
| `cleaning_date`           | 6             | 1.55%       |                                      |
| `substitution_reason`     | 364           | **94.06%**  | Datos extremadamente escasos         |
| `lab_observations`        | 336           | **86.82%**  | Campo poco documentado               |

**Acciones sugeridas**:  
- Eliminar columnas `substitution_reason` y `lab_observations` por alta tasa de faltantes (>85%).  
- Imputar valores en columnas con pocos faltantes (ej: `approver`).  

---

## Columnas Temporales Clave
- **Fechas críticas**:
  - `lab_received_date`: Recepción en laboratorio.
  - `scheduled_delivery_date`: Entrega programada.
  - `delivery_date`: Entrega real.
  - `calibration_date`: Fecha de calibración.

**Análisis sugerido**:  
- Calcular retrasos: `delivery_date - scheduled_delivery_date`.  
- Tiempo total de proceso: `delivery_date - lab_received_date`.  

---

## Variables Numéricas
- **Columnas**:
  - `delivery_time`: Tiempo de entrega.
  - `assigned_time`: Tiempo de asignación.
  - `piece_count`: Cantidad de piezas.

**Recomendaciones**:  
- Buscar **outliers** usando boxplots (ej: `delivery_time` anormalmente alto).  
- Analizar correlación entre `piece_count` y tiempos de proceso.  

---

## Variables Categóricas Clave
- **Personal**:
  - `assignee`, `calibrator`, `supervisor`: Evaluar carga de trabajo.  
- **Operacionales**:
  - `priority`: Prioridad del servicio ("URGENTE", "ORDINARIO").  
  - `service_location`: Ubicación del servicio ("LAB", "SITIO").  
  - `process_status`: Estado del proceso.
  

---

## Pasos Siguientes
1. **Limpieza de datos**:  
   - Eliminar columnas redundantes (`substitution_reason`, `lab_observations`).  
   - Validar consistencia temporal (ej: ¿`cleaning_date` ≤ `calibration_date`?).  

2. **Análisis de retrasos**:  
   - Crear columna `delay_days` para cuantificar incumplimientos.  

3. **Visualización**:  
   - Heatmap de correlación entre variables numéricas.  
   - Gráfico de barras apiladas para `process_status` por `priority`.  

4. **Optimización**:  
   - Identificar asignados (`assignee`) con mayor carga usando `piece_count` y `assigned_time`.  

**Objetivo final**: Mejorar la eficiencia operativa y reducir tiempos de entrega.  

# Análisis de los errores de sustitución


```python
# Eliminamos la columna 'service_type' y obtenemos la cantidad de reportes por cada razón de sustitución
#mass_cal = mass_cal.drop(columns=['service_type'], axis=1)
subs_reason = mass_cal.groupby('substitution_reason')['report_number'].count().sort_values(ascending=False)
subs_reason

```


```python
# Crear figura
plt.figure(figsize=(10, 6))

# Gráfico de barras con Seaborn
ax = sns.barplot(x=subs_reason.index, y=subs_reason.values, hue=subs_reason.index, palette='tab10', legend=False)

# Rotar etiquetas del eje X
plt.xticks(rotation=45)

total = mass_cal['report_number'].count()

# Etiquetas de los ejes
plt.xlabel('Razones de sustitución')
plt.ylabel('Catidad')
plt.title('Razones de sustitución por tipo de error en Masa', fontsize=16, fontweight='bold')

# Agregar etiquetas en las barras
for i, v in enumerate(subs_reason.values):
    ax.text(i, v + 0.1, str(v), ha='center', fontsize=12, fontweight='bold')
    
# Añadir porcentajes
for i, val in enumerate(subs_reason.values):
    porcentaje = val * 100 / total  # Calcula el porcentaje
    plt.text(
        i,                        # Posición X
        val - 0.5,                  # Posición Y (ajustada para centrar el texto)
        f"{porcentaje:.2f}%",     # Texto con 2 decimales
        ha='center',              # Alineación horizontal
        va='center',              # Alineación vertical (opcional)
        fontsize=10               # Tamaño de fuente (opcional)
    )

# Mostrar gráfico
plt.show()

```

#### Clasificación por metrólogo y tipos de errores


```python
# Agrupar por 'assignee' y contar errores por tipo de 'substitution_reason'
errores_por_assignee = (
    mass.groupby(['assignee', 'substitution_reason'])  # Agrupar por persona y tipo de error
        .size()                                   # Contar ocurrencias
        .unstack(fill_value=0)                    # Convertir a formato tabla
        .reset_index()                            # Convertir índice a columna
)

# Ordenar por el total de errores (opcional)
errores_por_assignee['Total'] = errores_por_assignee.count(axis=1)
errores_por_assignee = errores_por_assignee.sort_values('Total', ascending=False).drop('Total', axis=1)

# Mostrar resultado
errores_por_assignee
```


```python
# Configurar el estilo
plt.style.use('ggplot')

# Crear gráfico de barras apiladas
ax = errores_por_assignee.set_index('assignee').plot(
    kind='bar', 
    stacked=True,
    figsize=(14, 7),
    colormap='tab20',  # Paleta de colores para múltiples categorías
    edgecolor='black',
    linewidth=0.5
)

# Personalizar el gráfico
plt.title('Distribución de Errores por Asignado', fontsize=16, pad=20)
plt.xlabel('Asignado', fontsize=12)
plt.ylabel('Cantidad de Errores', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

# Añadir etiquetas de totales
for rect in ax.patches:
    height = rect.get_height()
    if height > 0:  # Mostrar solo etiquetas para barras con valores
        ax.text(rect.get_x() + rect.get_width()/2, 
                rect.get_y() + height/2, 
                f'{int(height)}', 
                ha='center', 
                va='center',
                fontsize=8,
                color='black')

# Mejorar la leyenda
plt.legend(
    title='Tipos de Error',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=9,
    frameon=True
)

# Ajustar márgenes
plt.tight_layout()
plt.show()
```

#### Tipos de errores vs Clasificación de equipo


```python
# 1. Crear una copia del DataFrame para preservar los datos originales
df_temp = mass_cal.copy()

# 2. Identificar errores no clasificados (NaN)
df_temp['classification'] = df_temp['classification'].fillna('No clasificado')  # Renombrar NaN

# 3. Agrupar incluyendo la nueva categoría
errores_por_classification = (
    df_temp.groupby(['classification', 'substitution_reason'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
)

# 4. Calcular el TOTAL correcto (suma de razones)
columnas_errores = errores_por_classification.columns.difference(['classification'])
errores_por_classification['Total'] = errores_por_classification[columnas_errores].sum(axis=1)

# 5. Ordenar y mostrar
errores_por_classification = errores_por_classification.sort_values('Total', ascending=False)
errores_por_classification
```


```python
# Configurar el estilo
plt.style.use('ggplot')

# Crear gráfico de barras apiladas
errores_por_classification = errores_por_classification.drop(columns='Total', errors='ignore')  # Eliminar la columna 'Total' para el gráfico
ax = errores_por_classification.set_index('classification').plot(
    kind='bar', 
    stacked=True,
    figsize=(14, 7),
    colormap='tab20',  # Paleta de colores para múltiples categorías
    edgecolor='black',
    linewidth=0.5
)

# Personalizar el gráfico
plt.title('Distribución de Errores por Clasificación', fontsize=16, pad=20)
plt.xlabel('Clasificación', fontsize=12)
plt.ylabel('Cantidad de Errores', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

# Añadir etiquetas de totales
for rect in ax.patches:
    height = rect.get_height()
    if height > 0:  # Mostrar solo etiquetas para barras con valores
        ax.text(rect.get_x() + rect.get_width()/2, 
                rect.get_y() + height/2, 
                f'{int(height)}', 
                ha='center', 
                va='center',
                fontsize=8,
                color='black')

# Mejorar la leyenda
plt.legend(
    title='Tipos de Error',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    fontsize=9,
    frameon=True
)

# Ajustar márgenes
plt.tight_layout()
plt.show()
```

#### Eliminar las filas que contienen los informes que corresponden a las sustituciones por errores


```python
mass_sin_errores = df_temp.drop(df_temp[df_temp['substitution_reason'].notna()].index)
```


```python
mass_sin_errores.loc[mass_sin_errores['report_number'] == 'CCM0095.M/2025', 'piece_count'] = 95
```


```python
mass_sin_errores = mass_sin_errores.drop(columns=['substitution_reason'], errors='ignore')
mass_sin_errores.head(10)
```


```python
# Eliminamos las columnas que no son necesarias
mass_sin_errores = mass_sin_errores.drop(columns=['substitution_reason', 'service_type', 'lab_observations', 'process_status'], errors='ignore')
```


```python
mass_sin_errores.info()
```

#### Comenzamos a trabajar sobre esta nueva tabla


```python
# Verificamos los valores nulos de la columna 'approver' en la tabla

resultado = pysqldf("""
    SELECT *
    FROM mass_sin_errores
    WHERE approver IS NULL """)

resultado
```


```python
#Cambiar el valor de la columna approver a 'Gabriel Gallardo Camacho' para el report_number específico
mass_sin_errores.loc[mass_sin_errores['report_number'].isin(['CCM0049.M/2025', 'CCM0050.M/2025']), 'approver'] = 'Gabriel Gallardo Camacho'
```


```python
# Verificamos los valores nulos de las columnas con valores faltantes en la tabla

resultado = pysqldf("""
    SELECT report_number, scheduled_delivery_date, cleaning_date, service_location
    FROM mass_sin_errores
    WHERE scheduled_delivery_date IS NULL or cleaning_date IS NULL """)

resultado
```

**Los valores faltantes son consistentes con los datos**

##### Buscamos si quedaron algunos valores duplicados


```python
no_dup = mass_sin_errores[mass_sin_errores.duplicated(keep=False)]
no_dup
```

**Lo podemos considerar como un solo certificado con dos piezas distintas**


```python
mass_sin_errores = mass_sin_errores.drop_duplicates()
```


```python
mass_sin_errores.loc[mass_sin_errores['report_number'] == 'CCM0058.M/2025', 'piece_count'] = 2
```

#### Verificar nuevamente la información de la tabla


```python
mass_sin_errores.info()
```


```python
metrologo_por_classification = (
    mass_sin_errores.groupby(['classification', 'assignee'])['report_number']
        .size()
        .unstack(fill_value=0)
        .reset_index()
)
'''
# Calcular el TOTAL correcto (suma de razones)
columnas_errores = metrologo_por_classification.columns.difference(['classification'])
metrologo_por_classification['Total'] = metrologo_por_classification[columnas_errores].sum(axis=1)

# Ordenar y mostrar
metologo_por_classification = metrologo_por_classification.sort_values('Total', ascending=False)'''
metrologo_por_classification
```

met_x_class_nt = metrologo_por_classification.drop(columns='Total', errors='ignore')
met_x_class_nt= met_x_class_nt.set_index('classification')
met_x_class_nt


```python
df_melted = metrologo_por_classification.melt(
    id_vars='classification', 
    var_name='Metrólogo', 
    value_name='Cantidad'
)

# Configurar el estilo
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# Crear el gráfico de barras agrupadas
barplot = sns.barplot(
    x='classification', 
    y='Cantidad', 
    hue='Metrólogo', 
    data=df_melted,
    palette='tab10',
    ci=None
)

# Personalización
plt.title('Distribución de Clasificaciones por Metrólogo', fontsize=16, pad=20)
plt.xlabel('Clasificación', fontsize=12)
plt.ylabel('Cantidad', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Añadir valores en las barras
for p in barplot.patches:
    if p.get_height() > 0:
        barplot.annotate(
            f'{int(p.get_height())}', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', 
            va='center', 
            xytext=(0, 5), 
            textcoords='offset points',
            fontsize=9
        )

# Mejorar la leyenda
plt.legend(
    title='Metrólogo',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    frameon=True
)

# Ajustar layout
plt.tight_layout()
plt.show()
```


```python
metrologo_por_tipo = (
    mass_sin_errores.groupby(['equipment_type', 'assignee'])['report_number']
        .size()
        .unstack(fill_value=0)
        .reset_index()
)

# 4. Calcular el TOTAL correcto (suma de razones)
columnas_errores = metrologo_por_tipo.columns.difference(['equipment_type'])
metrologo_por_tipo['Total'] = metrologo_por_tipo[columnas_errores].sum(axis=1)

# 5. Ordenar y mostrar
metologo_por_tipo = metrologo_por_tipo.sort_values('Total', ascending=False)
metrologo_por_tipo
```


```python
met_x_type_nt = metrologo_por_tipo.drop(columns='Total', errors='ignore')
met_x_type_nt= met_x_type_nt.set_index('equipment_type')
met_x_type_nt
```


```python
df_melted = met_x_type_nt.reset_index().melt(
    id_vars='equipment_type', 
    var_name='Metrólogo', 
    value_name='Cantidad'
)
#print(df_melted)

total = mass_sin_errores['report_number'].count()
# Configurar el estilo
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# Crear el gráfico de barras agrupadas
barplot = sns.barplot(
    x='equipment_type', 
    y='Cantidad', 
    hue='Metrólogo', 
    data=df_melted,
    palette='tab10',
    errorbar=None
)

# Personalización
plt.title('Distribución de Tipos de equipos por Metrólogo', fontsize=16, pad=20)
plt.xlabel('Tipo de equipo', fontsize=12)
plt.ylabel('Cantidad', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Añadir valores en las barras
for p in barplot.patches:
    if p.get_height() > 0:
        barplot.annotate(
            f'{int(p.get_height())}', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', 
            va='center', 
            xytext=(0, 5), 
            textcoords='offset points',
            fontsize=9
        )




# Mejorar la leyenda
plt.legend(
    title='Metrólogo',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    frameon=True
)

# Ajustar layout
plt.tight_layout()
plt.show()
```


```python
res = pysqldf("""
    SELECT equipment_type, COUNT(report_number) AS total_reportes
    FROM mass_sin_errores
    GROUP BY equipment_type """)
res
```


```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
ax = sns.barplot(
    x='total_reportes',
    y='equipment_type',
    data=res,
    palette=['#4CAF50', '#2196F3'],  # Paleta de colores personalizada
    saturation=0.9,                   # Intensidad del color
    #edgecolor='black',                # Borde de las barras
    #linewidth=1
)

# Personalización del título y etiquetas
plt.title('Cantidad de Reportes por Tipo de Equipo', 
         fontsize=14, pad=20, weight='semibold')
plt.xlabel('Número de Reportes', fontsize=12, labelpad=10)
plt.ylabel('')  # Eliminar etiqueta Y redundante

# Quitar bordes del gráfico
sns.despine(left=True, bottom=True)

# Añadir valores y porcentajes con mejor formato
total = res['total_reportes'].sum()
total = int(total)  # Convertir a entero para mostrar sin decimales
for p in ax.patches:
    width = p.get_width()
    percentage = width/total * 100
    ax.text(
        width + 3,                    # Posición X (valor + margen)
        p.get_y() + p.get_height()/2, # Posición Y centrada verticalmente
        f'{width}\n({percentage:.1f}%)', 
        va='center',
        ha='left',
        fontsize=10,
        color='#2d3436'
    )

# Ajustar límites del eje X
plt.xlim(0, res['total_reportes'].max() * 1.2)

# Mejorar formato de ejes
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
```


```python
df_melted = met_x_type_nt.reset_index().melt(
    id_vars='equipment_type', 
    var_name='Metrólogo', 
    value_name='Cantidad'
)

total = mass_sin_errores['report_number'].count()  # Asegúrate de que este sea el total correcto

# Configurar el estilo
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# Crear el gráfico de barras agrupadas
barplot = sns.barplot(
    x='equipment_type', 
    y='Cantidad', 
    hue='Metrólogo', 
    data=df_melted,
    palette='tab10',
    errorbar=None
)

# Personalización
plt.title('Distribución de Tipos de Equipos por Metrólogo', fontsize=16, pad=20)
plt.xlabel('Tipo de Equipo', fontsize=12)
plt.ylabel('Cantidad', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Añadir valores en las barras (PORCENTAJES Y CANTIDADES)
for p in barplot.patches:
    if p.get_height() > 0:
        # Calcular el porcentaje respecto al total
        porcentaje = (p.get_height() / total) * 100
        # Texto: cantidad + porcentaje
        texto = f"{int(p.get_height())}\n({porcentaje:.2f}%)"
        barplot.annotate(
            texto,
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 5),
            textcoords='offset points',
            fontsize=9
        )

# Mejorar la leyenda
plt.legend(
    title='Metrólogo',
    bbox_to_anchor=(1.05, 1),
    loc='upper left',
    frameon=True
)

# Ajustar layout
plt.tight_layout()
plt.show()
```


```python
class_del_time = mass_sin_errores.groupby(['classification'])['delivery_time'].mean().round(2).reset_index()
class_del_time
```


```python

plt.figure(figsize=(10, 6))
# Sintaxis básica
ax = sns.barplot(
    x='classification',  # Variable categórica (eje x)
    y='delivery_time',   # Variable numérica (eje y)
    data=class_del_time,
    #estimator='mean',  # Función de agregación (mean, sum, median, etc.)
    palette='magma',            # Intervalo de confianza (o 'sd' para desviación estándar)
    #color='blue'       # Color de las barras
)

# Añadir los valores en las barras
for bar in ax.patches:
    height = bar.get_height()
    ax.text(
        x=bar.get_x() + bar.get_width() / 2,  # Posición horizontal centrada
        y=height + 0.2,                       # Posición vertical (altura + offset)
        s=f"{height:.2f}",                    # Texto (2 decimales)
        ha="center",                           # Alineación horizontal
        va="bottom",                           # Alineación vertical
        fontsize=10
    )

plt.title("Promedio de tiempo por clasificación de equipo", fontsize=14)
plt.xlabel("Clasificación", fontsize=12)
plt.ylabel("Tiempo promedio", fontsize=12)
plt.xticks(rotation=45, ha='left', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Solo líneas de cuadrícula en el eje y

plt.show()
```


```python
tiempos = pysqldf('''
    SELECT classification, AVG(delivery_time), AVG(assigned_time)
    FROM mass_sin_errores
    GROUP BY classification
''')
tiempos
```


```python
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar el estilo
equipos = tiempos['classification'].tolist()
asignado = tiempos['AVG(assigned_time)'].tolist()
real = tiempos['AVG(delivery_time)'].tolist()

plt.figure(figsize=(14, 6))
ax = sns.barplot(x=equipos, y=asignado, color='#4CAF00', label='Tiempo Asignado')
sns.barplot(x=equipos, y=real, color='#1191F3', label='Tiempo de Entrega Real', alpha=0.7)

# Personalización
plt.title('Comparación de Tiempos Asignados vs Reales por Clasificación', pad=20)
plt.xlabel('Clasificaciones')
plt.ylabel('Días')
plt.xticks(rotation=45)
plt.legend()

# Añadir valores
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 5), 
                textcoords='offset points')

plt.tight_layout()
plt.show()
```


```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x=asignado, y=real, hue=equipos, s=150, palette='viridis')

# Línea de referencia ideal
max_val = max(max(asignado), max(real)) + 5
plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

# Personalización
plt.title('Relación Tiempos Asignados vs Reales', pad=15)
plt.xlabel('Tiempo Asignado Promedio (días)')
plt.ylabel('Tiempo Real Promedio (días)')
plt.grid(alpha=0.3)

# Añadir etiquetas
for i, txt in enumerate(equipos):
    plt.annotate(txt, (asignado[i]+0.5, real[i]))

plt.tight_layout()
plt.show()
```


```python
diferencias = [r - a for r, a in zip(real, asignado)]

plt.figure(figsize=(12, 6))
bars = plt.barh(equipos, diferencias, color=np.where(np.array(diferencias) > 0, '#e74c3c', '#2ecc71'))

# Personalización
plt.title('Diferencia entre Tiempo Real y Asignado', pad=15)
plt.xlabel('Diferencia (Real - Asignado) en Horas')
plt.grid(axis='x', alpha=0.3)

# Añadir valores
for bar in bars:
    width = bar.get_width()
    plt.text(width/2, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}', 
             va='center', 
             color='white' if abs(width) > 1 else 'black')

plt.tight_layout()
plt.show()
```

# Para otro informe


```python
recibidos = mass_sin_errores.groupby('lab_received_date')['report_number'].count().reset_index()
entregados = mass_sin_errores.groupby('delivery_date ')['report_number'].count().reset_index()
recibidos = recibidos.rename(columns={'report_number': 'recibidos'})
entregados = entregados.rename(columns={'report_number': 'entregados'})
recibidos
```


```python
time_analisys_lab = mass_sin_errores.loc[mass_sin_errores['service_location'] == 'LAB']
#time_analisys = pd.merge(recibidos, entregados, left_on='lab_received_date', right_on='delivery_date ')
time_analisys_lab
```


```python
time_analisys_lab.loc[time_analisys_lab['lab_received_date'] < '2025-01-01', 'lab_received_date'] = '2025-01-01'
```


```python
recibidos = time_analisys_lab.groupby('lab_received_date')['report_number'].count().reset_index()
entregados = time_analisys_lab.groupby('delivery_date ')['report_number'].count().reset_index()
```

recibidos.rename(columns={'report_number': 'recibidos'})
entregados.rename(columns={'report_number': 'entregados'})


```python
resultado =pd.merge(recibidos, entregados, left_on='lab_received_date', right_on='delivery_date ', how='outer')
resultado.rename(columns={'report_number_x': 'recibidos', 'report_number_y': 'entregados'}, inplace=True)
resultado = resultado.fillna({'recibidos': 0, 'entregados': 0})
#

#resultado['diferencia'] = resultado['recibidos'] - resultado['entregados']
resultado
```


```python
resultado.fillna({'lab_received_date': resultado['delivery_date '], 'delivery_date ': resultado['lab_received_date']}, inplace=True)

```


```python
resultado.drop(columns=['delivery_date '], inplace=True, axis=1)
```


```python
resultado.rename(columns={'lab_received_date': 'fecha'}, inplace=True)
resultado['diferencia'] = (resultado['recibidos'] - resultado['entregados']).cumsum()
#resultado['carga'] = resultado['diferencia'].cumsum()
resultado
```


```python
carga = resultado[['fecha', 'diferencia']]
carga
```


```python


# Configurar el estilo
plt.figure(figsize=(14, 6))
plt.style.use('ggplot')  # Estilo profesional

# Crear la gráfica de línea
plt.plot(
    carga['fecha'],
    carga['diferencia'],
    marker='o',                # Marcadores en cada punto
    linestyle='--',            # Línea discontinua
    color='#2ecc71',           # Color verde
    linewidth=2,
    markersize=8
)

# Personalización
plt.title('Serie de Tiempo: Diferencia Diaria (Recibidos - Entregados)', 
         fontsize=14, pad=20)
plt.xlabel('Fecha', fontsize=12, labelpad=10)
plt.ylabel('Diferencia', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right')  # Rotar fechas para mejor legibilidad

# Línea de referencia en cero
plt.axhline(0, color='#e74c3c', linestyle='-', linewidth=1, alpha=0.7)

# Añadir cuadrícula
plt.grid(axis='y', alpha=0.3)

# Ajustar márgenes
plt.tight_layout()
plt.show()
```


```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Convertir 'fecha' a datetime y establecer como índice
carga['fecha'] = pd.to_datetime(carga['fecha'])
carga.set_index('fecha', inplace=True)

# Agrupar por semana (suma de diferencias)
semanal = carga.resample('W-MON')['diferencia'].sum().reset_index()

# Configurar el gráfico
plt.figure(figsize=(14, 6))
plt.fill_between(
    semanal['fecha'],
    semanal['diferencia'],
    color='#3498db',
    alpha=0.4,
    linewidth=2,
    edgecolor='#2c3e50'
)

# Personalizar eje x para semanas
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n(Semana %W'))  # Formato: 01-Ene (Semana 01)
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY, interval=1))  # Marcas cada lunes

# Añadir líneas verticales para semanas
plt.grid(axis='x', color='gray', linestyle='--', alpha=0.3)

# Títulos y estilo
plt.title('Diferencia Semanal: Equipos Recibidos vs Entregados', fontsize=14, pad=20)
plt.xlabel('Semana', fontsize=12, labelpad=15)
plt.ylabel('Diferencia Acumulada', fontsize=12, labelpad=10)
plt.xticks(rotation=45, ha='right', fontsize=10)

# Línea de referencia en cero
plt.axhline(0, color='#e74c3c', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()

```


```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Convertir a datetime si no está ya en este formato
carga['fecha'] = pd.to_datetime(carga['fecha'])

# Crear figura
plt.figure(figsize=(16, 6))

# Gráfico de área
plt.fill_between(
    carga['fecha'],
    carga['diferencia'],
    color='#3498db',
    alpha=0.4,
    linewidth=2,
    edgecolor='#2c3e50'
)

# Personalización del eje x
ax = plt.gca()

# Formato de fechas
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))  # Formato: 05-Ene
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))      # Mostrar cada 2 días

# Rotación y alineación
plt.xticks(rotation=45, ha='right', fontsize=10)

# Añadir fechas menores para mejor resolución
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))

# Ajustar márgenes
plt.margins(x=0.01)  # Reducir espacio en los extremos

# Títulos y estilo
plt.title('Serie Temporal de Diferencia Diaria', fontsize=14, pad=20)
plt.xlabel('Fecha', fontsize=12, labelpad=15)
plt.ylabel('Diferencia (Recibidos - Entregados)', fontsize=12, labelpad=10)
plt.grid(axis='x', alpha=0.3, linestyle='--')

# Línea de referencia en cero
plt.axhline(0, color='#e74c3c', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()
```

# Para entregar


```python
tiempo_gral = mass_sin_errores.loc[mass_sin_errores['service_location'] == 'LAB'][['report_number','classification', 'lab_received_date','assigned_time', 'delivery_date ','delivery_time', ]].copy()
```


```python
tiempo_gral['efectividad'] = (tiempo_gral['delivery_time']/tiempo_gral['assigned_time']).round(2)
tiempo_gral
salva = tiempo_gral.copy()
```


```python
tiempo_gral.rename(columns={'delivery_date ': 'delivery_date'}, inplace=True)
```


```python
# Convertir a datetime y crear columnas de periodo
tiempo_gral['delivery_date'] = pd.to_datetime(tiempo_gral['delivery_date'])
tiempo_gral['semana'] = tiempo_gral['delivery_date'].dt.to_period('W-MON')  # Semana que comienza en lunes
tiempo_gral['mes'] = tiempo_gral['delivery_date'].dt.to_period('M')
```


```python
tiempo_gral
```


```python
tiempo_gral.columns
```


```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Convertir columnas clave a formato correcto
tiempo_gral = tiempo_gral.assign(
    lab_received_date = pd.to_datetime(tiempo_gral['lab_received_date']),
    assigned_time = pd.to_numeric(tiempo_gral['assigned_time'], errors='coerce'),
    delivery_date = pd.to_datetime(tiempo_gral['delivery_date']),
    delivery_time = pd.to_numeric(tiempo_gral['delivery_time'], errors='coerce'),
    
    # Convertir columnas de período a datetime correctamente
    #semana = tiempo_gral['semana'].dt.to_timestamp(),  # <-- Solución clave
   # mes = tiempo_gral['mes'].dt.to_timestamp()         # <-- Solución clave
)

# Eliminar filas con valores inválidos
tiempo_gral = tiempo_gral.dropna(subset=['delivery_time', 'assigned_time'])

# Calcular efectividad (si es necesario)
#tiempo_gral['efectividad'] = tiempo_gral['delivery_time'] / tiempo_gral['assigned_time']

# Verificar tipos de datos
print(tiempo_gral[['semana', 'mes']].dtypes)
```


```python
# Agrupamiento semanal
semanal_general = tiempo_gral.resample('W-MON', on='semana')['delivery_time'].mean()

# Agrupamiento mensual
mensual_general = tiempo_gral.resample('M', on='mes')['delivery_time'].mean()

# Visualización
fig, ax = plt.subplots(2, 1, figsize=(14, 8))

# Gráfico semanal
sns.lineplot(
    x=semanal_general.index,
    y=semanal_general.values,
    ax=ax[0],
    marker='o',
    color='#2ecc71'
)
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b\n(Semana %W)'))
ax[0].set_title('Tiempo de Entrega Promedio Semanal', fontsize=12)

# Gráfico mensual
sns.lineplot(
    x=mensual_general.index,
    y=mensual_general.values,
    ax=ax[1],
    marker='o',
    color='#e74c3c'
)

ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
ax[1].set_title('Tiempo de Entrega Promedio Mensual', fontsize=12)

# Rotar etiquetas y personalizar
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.title('Tiempo de Entrega Promedio Mensual', fontsize=14, pad=20)
plt.ylabel('Horas', fontsize=12)
plt.xlabel('Mes', fontsize=12)

plt.tight_layout()
plt.show()
```


```python

```
