import pandas as pd
import os
from typing import List, Tuple
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def to_prepare_db(db: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa el DataFrame:
    - Une el contenido de 'Asignados' de las filas con 'NUMERO DE INFORME' NaN 
      a la fila anterior.
    - Elimina las filas donde 'NUMERO DE INFORME' es NaN.
    
    Args:
        db (pd.DataFrame): DataFrame de entrada a procesar
    
    Returns:
        pd.DataFrame: DataFrame procesado
    """
    try:
        # Validar que la entrada sea un DataFrame
        if not isinstance(db, pd.DataFrame):
            raise TypeError("La entrada debe ser un DataFrame de pandas")
        
        # Validar columnas requeridas
        columnas_requeridas = ['NUMERO DE INFORME', 'Asignados']
        for columna in columnas_requeridas:
            if columna not in db.columns:
                raise ValueError(f"Columna '{columna}' no encontrada en el DataFrame")
        
        # Crear copia del DataFrame para evitar modificar el original
        db_copy = db.copy()
        
        # Identificar filas con NaN en 'NUMERO DE INFORME'
        sin_inf = db_copy[db_copy['NUMERO DE INFORME'].isnull()].copy()
        
        # Recorrer índices en orden inverso para concatenar 'Asignados'
        for idx in reversed(sin_inf.index):
            fila_anterior = idx - 1
            
            # Verificar que la fila anterior existe
            if fila_anterior in db_copy.index:
                db_copy.at[fila_anterior, 'Asignados'] = (
                    f"{db_copy.at[fila_anterior, 'Asignados']}, {db_copy.at[idx, 'Asignados']}"
                )
        
        # Eliminar filas sin número de informe
        prep_db = db_copy.dropna(subset=['NUMERO DE INFORME']).reset_index(drop=True)
        
        return prep_db
    
    except Exception as e:
        logging.error(f"Error al preparar el DataFrame: {e}")
        raise

def to_store() -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Solicita y procesa archivos CSV.
    
    Returns:
        Tuple con lista de DataFrames procesados y sus nombres de salida
    """
    nombres = []
    
    # Solicitar nombres de archivos
    while True:
        nombre = input("Ingresa un nombre de archivo CSV (sin extensión, Enter para salir): ").strip()
        
        if not nombre:
            break
        
        # Agregar extensión .csv
        nombre_archivo = f"{nombre}.csv"
        
        # Validar existencia del archivo
        if not os.path.exists(nombre_archivo):
            logging.warning(f"El archivo '{nombre_archivo}' no existe. Se omitirá.")
            continue
        
        nombres.append(nombre_archivo)
    
    # Procesar archivos y generar nombres de salida
    try:
        dataframes = [to_prepare_db(pd.read_csv(nombre)) for nombre in nombres]
        nombres_salida = [nombre.replace('.csv', '_prep.csv') for nombre in nombres]
        
        return dataframes, nombres_salida
    
    except Exception as e:
        logging.error(f"Error procesando archivos: {e}")
        return [], []

def convert_to_csv():
    """
    Convierte DataFrames procesados a archivos CSV.
    Maneja posibles errores durante la conversión.
    """
    try:
        dataframes, nombres_salida = to_store()
        
        # Guardar DataFrames procesados
        for i, df in enumerate(dataframes):
            try:
                df.to_csv(nombres_salida[i], index=False)
                logging.info(f"Archivo guardado: {nombres_salida[i]}")
            except IOError as e:
                logging.error(f"Error guardando {nombres_salida[i]}: {e}")
        
        print("Archivos CSV creados exitosamente.")
    
    except Exception as e:
        logging.error(f"Error general en la conversión: {e}")

if __name__ == "__main__":
    convert_to_csv()
