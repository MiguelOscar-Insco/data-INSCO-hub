import pandas as pd
import os
from typing import List, Tuple
import logging
import sys

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def to_prepare_db(db: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa el DataFrame:
    - Une el contenido de 'assignee' de las filas con 'report_number' NaN 
      a la fila anterior.
    - Elimina las filas donde 'report_number' es NaN.
    
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
        columnas_requeridas = ['report_number', 'assignee']
        for columna in columnas_requeridas:
            if columna not in db.columns:
                raise ValueError(f"Columna '{columna}' no encontrada en el DataFrame")
        
        # Crear copia del DataFrame para evitar modificar el original
        db_copy = db.copy()
        
        # Identificar filas con NaN en 'report_number'
        sin_inf = db_copy[db_copy['report_number'].isnull()].copy()
        
        # Recorrer índices en orden inverso para concatenar 'assignee'
        for idx in reversed(sin_inf.index):
            fila_anterior = idx - 1
            
            # Verificar que la fila anterior existe
            if fila_anterior in db_copy.index:
                db_copy.at[fila_anterior, 'assignee'] = (
                    f"{db_copy.at[fila_anterior, 'assignee']}, {db_copy.at[idx, 'assignee']}"
                )
        
        # Eliminar filas sin número de informe
        prep_db = db_copy.dropna(subset=['report_number']).reset_index(drop=True)
        
        return prep_db
    
    except Exception as e:
        logging.error(f"Error al preparar el DataFrame: {e}")
        raise

def cargar_archivo(nombre_archivo: str) -> pd.DataFrame:
    """
    Carga un archivo CSV en un DataFrame.
    
    Args:
        nombre_archivo (str): Nombre del archivo a cargar
        
    Returns:
        pd.DataFrame: DataFrame con los datos del archivo
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        Exception: Para otros errores en la carga
    """
    try:
        if not os.path.exists(nombre_archivo):
            raise FileNotFoundError(f"El archivo '{nombre_archivo}' no existe")
            
        return pd.read_csv(nombre_archivo)
    except Exception as e:
        logging.error(f"Error al cargar el archivo {nombre_archivo}: {e}")
        raise

def guardar_archivo(df: pd.DataFrame, nombre_salida: str) -> bool:
    """
    Guarda un DataFrame en un archivo CSV.
    
    Args:
        df (pd.DataFrame): DataFrame a guardar
        nombre_salida (str): Nombre del archivo de salida
        
    Returns:
        bool: True si se guardó correctamente, False en caso contrario
    """
    try:
        df.to_csv(nombre_salida, index=False)
        logging.info(f"Archivo guardado: {nombre_salida}")
        return True
    except Exception as e:
        logging.error(f"Error guardando {nombre_salida}: {e}")
        return False

def procesar_archivos() -> None:
    """
    Solicita nombres de archivos al usuario, los procesa y guarda los resultados.
    """
    try:
        # Lista para almacenar nombres de archivos a procesar
        archivos_a_procesar = []
        
        # Solicitar nombres de archivos
        print("Ingresa un nombre de archivo CSV (sin extensión, Enter para salir):")
        while True:
            nombre = input().strip()
            
            if not nombre:
                break
            
            # Agregar extensión .csv si no la tiene
            if not nombre.lower().endswith('.csv'):
                nombre_archivo = f"{nombre}.csv"
            else:
                nombre_archivo = nombre
            
            archivos_a_procesar.append(nombre_archivo)
        
        if not archivos_a_procesar:
            print("No se ingresaron archivos para procesar.")
            return
            
        archivos_procesados = 0
        
        # Procesar cada archivo
        for archivo in archivos_a_procesar:
            try:
                # Generar nombre de archivo de salida
                nombre_salida = archivo.replace('.csv', '_prep.csv')
                
                # Cargar, procesar y guardar
                df = cargar_archivo(archivo)
                df_procesado = to_prepare_db(df)
                if guardar_archivo(df_procesado, nombre_salida):
                    archivos_procesados += 1
                    
            except FileNotFoundError as e:
                logging.warning(str(e))
            except Exception as e:
                logging.error(f"Error procesando {archivo}: {e}")
        
        if archivos_procesados > 0:
            print(f"Archivos CSV creados exitosamente: {archivos_procesados}")
        else:
            print("No se pudo procesar ningún archivo.")
            
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
    except Exception as e:
        logging.error(f"Error general en el procesamiento: {e}")

if __name__ == "__main__":
    procesar_archivos()
    