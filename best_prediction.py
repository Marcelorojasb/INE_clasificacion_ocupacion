# El siguiente archivo permite consolidar un CSV que contega la mejor predicción posible
# comparando las predicciones por clase de cada modelo y almacenando la "mejor predicción"

import os
import pandas as pd

def consolidar_clasificaciones(carpeta):
    # Obtener la lista de archivos CSV en el directorio
    archivos_csv = [archivo for archivo in os.listdir(carpeta) if archivo.endswith('.csv')]

    # Verificar si hay archivos CSV en el directorio
    if not archivos_csv:
        print("No se encontraron archivos CSV en el directorio proporcionado.")
        return

    # Inicializar un diccionario para almacenar los datos consolidados
    datos_consolidados = {}

    # Recorrer cada archivo CSV y consolidar los datos
    for archivo_csv in archivos_csv:
        ruta_archivo = os.path.join(carpeta, archivo_csv)
        # Leer el archivo CSV
        datos = pd.read_csv(ruta_archivo)
        # Iterar sobre cada fila del DataFrame
        for indice, fila in datos.iterrows():
            id_texto = fila['id']
            clase = fila['preds']
            probabilidad = fila['probs']
            modelo = archivo_csv

            # Verificar si el ID de texto ya está en el diccionario
            if id_texto in datos_consolidados:
                # Verificar si la probabilidad es mayor que la almacenada
                if probabilidad > datos_consolidados[id_texto]['prob']:
                    # Actualizar los datos consolidados
                    datos_consolidados[id_texto] = {'clase': clase, 'prob': probabilidad, 'modelo': modelo}
            else:
                # Agregar el ID de texto al diccionario
                datos_consolidados[id_texto] = {'clase': clase, 'prob': probabilidad, 'modelo': modelo}

    # Convertir el diccionario a un DataFrame
    df_consolidado = pd.DataFrame.from_dict(datos_consolidados, orient='index').reset_index()
    df_consolidado.columns = ['ID', 'clase', 'prob', 'modelo']

    # Obtener el nombre del modelo de la ruta
    nombre_modelo = os.path.basename(carpeta)

    # Guardar el DataFrame consolidado en un archivo CSV
    ruta_salida = os.path.join("data_out", f'clasificacion_{nombre_modelo}.csv')
    df_consolidado.to_csv(ruta_salida, index=False)

    print(f"Clasificación consolidada guardada en: {ruta_salida}")
