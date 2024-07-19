# Modelo para Clasificación de Ocupación a partir de la Encuesta Nacional de Empleo
----------------------

El siguiente repositorio tiene la finalidad de facilitar la clasificación ocupación laboral a partir de las respuestas de la Encuesta Nacional de Empleo.

## ¿Qué hace este repositorio?
El proyecto utiliza tres modelos (BERT, SVM y MLP) para realizar la clasificación de forma independiente para el caso de cada clasificador (Caenes y Ciuo-08). 
El modelo está construido como una combinación de modelos independientes que llevan a cabo la clasificación de forma simultánea. Posteriormente, y en base a las métricas obtenidas de la clasificación, se genera un resultado consolidado el cual agrupa los mejores resultados de clasificación para cada clase en un archivo CSV dentro de la carpeta *data_out*.

## Información general
Es repositorio cuenta con cinco carpetas principales y tres archivos "sueltos" importantes.
### Carpetas
- **data_in**: Esta carpeta almacena todos los "inputs" del modelo. Para actualizar los datos a clasificar, deberán reemplazarse los archivos ".csv" existentes en la carpeta (si es que existen). Es importante que siempre se guarden con el mismo nombre, es decir, "caenes.csv" y "ciuo08_v8.csv". El archivo "build_data.py" almacena la información necesaria para construir los set de datos necesario para la clasificación a partir de los archivos CSV cargados, por lo que **no debe** ser alterado.
- **data_out**: En esta carpeta se guardan los archivos CSV que contienen los resultados de las clasificaciones los modelos. Además, se pueden encontrar cuatro carpetas correspondiente a cada clasificador, donde se almacena el archivo consolidado con las mejores predicciones por clase. Antes de llevar a cabo una clasificación, recomendamos vaciar este directorio cuidando no eliminar las carpetas existentes, mas sí los archivos dentro de ellas.
- **models**: Aquí se almacenan los modelos necesarios para realizar la clasificación. Dada alta cantidad de información que contienen los archivos de los modelos, no han podido ser guardados directamente en este repositorio. Deberás ingresar al siguiente [link](https://drive.google.com/drive/folders/1VbCUkdXT2jEeQgcpO-TRQToggyogHj0g?usp=sharing) para poder descargar los modelos y, a continuación, cargarlos en esta carpeta. El archivo "bert.py" que vive en este directorio tiene la función de procesar los datos para que puedan ser trabajados por el modelo BERT, por lo que **tampoco se debe alterar**.
- **predictors**: Los archivos almacenados en esta carpeta crean las predicciones de la clasificación para los modelos cargados y los almacenan en la carpeta "data_out". **No debe ser modificados**.
- **utils**: Aquí se almacena el archivo "metrics.py", el cual permite obtener las métricas de comparación entre los modelos.
### Archivos importantes
- **main.py**: Este archivo almacena toda la información necesaria para llevar a cabo la clasificación. Es el archivo a ejecutar cuando se quiera llevar a cabo una nueva implementación.
- **best_prediction.py**: Este contiene una función indispensable para la creación de los archivos consolidados, la cual es referenciada en el archivo "main.py" para que al ser ejecutado se almacenen los resultados finales en "data_out".
- **requirementes.txt**: El archivo de requerimientos contienen la información de las librerías y paquetes necesarios para le funcionamiento del repositorio. Es importante instalar estos requerimientos en un entorno virtual antes de ejecutar el archivo "main.py".
### Otros
Podrán ver la existencia de dos archivos con la extensión ".ipynb", estos archivos son Jupyter Notebooks que fueron utilizados para validar los modelos creados. Son dispensables para el funcionamiento del repositorio.

## Instrucciones de uso
1. Descarga este repositorio o clónalo dentro de una carpeta de tu directorio de trabajo
      - Para clonar este repositorio deberás tener instalado Git. En caso de no tener Git en tu dispositivo, puedes descargarlo desde el siguiente [link](https://git-scm.com/download/win).
      - Una vez instalado Git, deberás abrir la aplicación Git Bash. Esto puedes hacerlo apretando clic derecho dentro de la carpeta en que se quiera clonar el repositorio y seleccionando "Abrir Git Bash aquí"
      - Dentro de Git Bash, deberán ingresar el siguiente comando: `git clone https://github.com/Osilva97/P13-INE.git`
      - Debería poder ver el repositorio en la carpeta donde fue clonado y navegar por sus archivos
2. Crea un entorno virtual e instala el archivo con los requerimientos del proyecto
      - **IMPORTANTE**: El repositorio funciona con la versión de **Python 3.10.12** (la versión 3.10.11 puede ser compatible), por lo que asegúrate de que crees tu entorno virtual con dicha versión. Puedes verificar la versión de python antes de crear el entorno virtual ingresando el siguiente comando: `python --version`. Si tu versión de python es distinta, debes instalar la versión indicada antes de realizar los pasos siguientes. Puedes descargar la versión indicada desde el siguiente [link](https://www.python.org/downloads/release/python-31012/).
      - Abre la consola de comandos de windows y navega hasta la carpeta donde almacenaste el repositorio
      - Crea un entorno virtual ingresando el siguiente comando: `python -m venv <<nombre_entorno>>`
      - Verifica si el entorno está activado, debe aparecer *(nombre_entorno)* antepuesto a la ruta de su directorio la consola de comandos
      - Si no se encuentra ativado, activa el entorno virual ingresando el siguiente comando: `<<nombre_entorno>>\Scritpts\activate`
      - Instalar dependencias del proyecto en el entorno virual ingresando el siguiente comando: `pip install -r requirements.txt`
      - El repositorio debería estar listo para su uso
4. Actualiza las dependencia de la carpeta "data_in". Basta con eliminar los archivos CSV existentes y cargar los archivos nuevos. Recuerde que estos deben mantener los nombres de los archivos originales, es decir: "caenes.csv" y "ciuo08_v8.csv".
5. Asegurate de que no haya información dentro de "data_out" además de las carpetas "caenesd1", "caenesd2", "ciuo1d" y "ciuo2d".
6. Carga los modelos de clasificación a la carpeta "models". Los modelos puedes encontrarlos en el siguiente [drive](https://drive.google.com/drive/folders/1VbCUkdXT2jEeQgcpO-TRQToggyogHj0g?usp=sharing).
7. En la consola de comandos, con el entorno virtual activado, ejecuta el archivo "main.py" ingresando el siguiente comando: `python main.py`
8. Pasado el tiempo de compilación, deberías ver los resultados en forma de archivos CSV en la carpeta "data_out".

## Links útiles:
- Creación de entornos virtuales: https://docs.python.org/es/3.10/library/venv.html
- Python 3.10.12 (descargar archivo tar): https://www.python.org/downloads/release/python-31012/
- Instalación de versiones antiguas de python desde archivos tar: https://medium.com/@lupiel/installing-python-from-a-tgz-file-a-step-by-step-guide-4cf5f4a17a86
