# El siguiente archivo contiene todo lo necesario para entrenar los modelos y obtener las predicciones necesarias
# Para correr este archivo por PRIMERA vez, debe hacer lo siguiente:
#      1. Abrir CMD de Windows
#      2. Navegar hasta el directorio del proyecto
#      3. Crear un entorno virtual de python:
#               Esto puede hacerlo con la siguiente línea de comando: python -m venv "nombre_entorno"
#      4. Verifique si el entorno está activado, debe aparecer "(nombre_entorno)" al inicio de su directorio en el shell de CMD
#      5. Si no se encuentra ativado, active el entorno virual:
#               Ingrese el comando: "nombre_entorno"\Scritpts\activate
#      6. Instalar dependencias del proyecto en el entorno virual:
#               Ingrese como comando: pip install -r requirements.txt
#      7. Ejecutar el archivo mediante la siguiente línea de comando: python main.py

# Importación de archivos internos
from models.bert import BERT_CIUOClass, BERT_CAENESClass
from predictors.bert import predict_single_sample, validation
from predictors.baselines import predict_baseline
from data_in.build_data import build_dataloader
from utils.metrics import calculate_metrics, plot_comparison_bar_graph, compare_models

# Importación de librerías necesarias
import torch
import pandas as pd
from transformers import BertTokenizer, BertModel, BertConfig
import unidecode
from spanish_nlp import preprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from pathlib import Path

sp = preprocess.SpanishPreprocess(
        lower=True,
        remove_url=False,
        remove_hashtags=False,
        split_hashtags=False,
        normalize_breaklines=True,
        remove_emoticons=False,
        remove_emojis=False,
        convert_emoticons=False,
        convert_emojis=False,
        normalize_inclusive_language=False,
        reduce_spam=False,
        remove_reduplications=False,
        remove_vowels_accents=False,
        remove_multiple_spaces=True,
        remove_punctuation=True,
        remove_unprintable=False,
        remove_numbers=False,
        remove_stopwords=True,
        stopwords_list='default',
        lemmatize=False,
        stem=False,
        remove_html_tags=False,
)

print('Cargando datos...')

ciuo_df = pd.read_csv("data_in/ciuo08_v8.csv",encoding='utf-8', index_col=0)
#ciuo_df = ciuo_df.drop(columns=['Unnamed: 0','b16_otro', 'cise', 'b16', 'curso', 'nivel', 'termino','nummesesempleo'])
ciuo_df.dropna(inplace=True)
ciuo_df = ciuo_df.drop(ciuo_df[ciuo_df.clase1 == 999].index)
ciuo_df = ciuo_df.drop(ciuo_df[ciuo_df.clase == 999].index)
ciuo_df['texto_ciuo'] = ciuo_df['texto_ciuo'].apply(lambda texto: sp.transform(texto, debug = False))
train_loader_ciuo1d, test_loader_ciuo1d, labels_unique_ciuo1d, test_dataset_ciuo1d, ciuo1d_X_train, ciuo1d_X_test, ciuo1d_y_train, ciuo1d_y_test = build_dataloader(ciuo_df, text='texto_ciuo',labels='clase1')
train_loader_ciuo2d, test_loader_ciuo2d, labels_unique_ciuo2d, test_dataset_ciuo2d, ciuo2d_X_train, ciuo2d_X_test, ciuo2d_y_train, ciuo2d_y_test = build_dataloader(ciuo_df, text='texto_ciuo',labels='clase')

# Dividimos el dataset en train y test, aún no se transforma de Strings a valores numéricos.
caenes_df = pd.read_csv("data_in/caenes.csv",encoding='latin1',index_col=0)
caenes_df.dropna(inplace=True)
caenes_df = caenes_df.drop(caenes_df[caenes_df.b14_1_rev4cl_caenes == 99].index)
caenes_df = caenes_df.drop(caenes_df[caenes_df.b14_2d == 99].index)
#caenes_df['texto'] = caenes_df['texto'].apply(lambda texto: sp.transform(texto, debug = False))
train_loader_caenes1d, test_loader_caenes1d, labels_unique_caenes1d, test_dataset_caenes1d, caenes1d_X_train, caenes1d_X_test, caenes1d_y_train, caenes1d_y_test = build_dataloader(caenes_df, text='texto',labels='b14_1_rev4cl_caenes')
train_loader_caenes2d, test_loader_caenes2d, labels_unique_caenes2d, test_dataset_caenes2d, caenes2d_X_train, caenes2d_X_test, caenes2d_y_train, caenes2d_y_test  = build_dataloader(caenes_df, text='texto',labels='b14_2d')
# Dividimos el dataset en train y test, aún no se transforma de Strings a valores numéricos.

print('Datos cargados.')

print('Inicializando modelos...')
# CIUO MODELS
model_ciuo1d = BERT_CIUOClass(out_size=10)
model_ciuo1d.load_state_dict(torch.load('models/BERT_ciuo_1d.pth',map_location=torch.device('cpu')))
model_ciuo2d = BERT_CIUOClass(out_size=42)
model_ciuo2d.load_state_dict(torch.load('models/BERT_ciuo_2d.pth',map_location=torch.device('cpu')))

# CAENES MODELS
model_caenes1d = BERT_CAENESClass(out_size=21)
model_caenes1d.load_state_dict(torch.load('models/BERT_caenes_1d.pth',map_location=torch.device('cpu')))
model_caenes2d = BERT_CAENESClass(out_size=81)
model_caenes2d.load_state_dict(torch.load('models/BERT_caenes_2d.pth',map_location=torch.device('cpu')))

print('Modelos cargados.')

print('Calculando predicciones...')
y_pred_ciuo1d = predict_baseline(ciuo1d_X_test,carpeta = 'ciuo1d',versions=['SVM_ciuo1d', 'MLP_ciuo1d'])
y_pred_ciuo2d = predict_baseline(ciuo2d_X_test,carpeta = 'ciuo2d',versions=['SVM_ciuo2d', 'MLP_ciuo2d'])
print('Baseline ciuo listo.')
y_pred_caenes1d = predict_baseline(caenes1d_X_test,carpeta = 'caenes1d',versions=['SVM_caenes1d', 'MLP_caenes1d'])
y_pred_caenes2d = predict_baseline(caenes2d_X_test,carpeta = 'caenes2d',versions=['SVM_caenes2d', 'MLP_caenes2d'])
print('Baseline caenes listo.')
fin_outputs_ciuo1d, fin_targets_ciuo1d = validation(model_ciuo1d, test_loader_ciuo1d)
fin_outputs_ciuo2d, fin_targets_ciuo2d = validation(model_ciuo2d, test_loader_ciuo2d)
print('Bert ciuo listo.')
fin_outputs_caenes1d, fin_targets_caenes1d = validation(model_caenes1d, test_loader_caenes1d)
fin_outputs_caenes2d, fin_targets_caenes2d = validation(model_caenes2d, test_loader_caenes2d)
print('Bert caenes listo.')

outputs_ciuo1d, targets_ciuo1d = np.argmax(fin_outputs_ciuo1d, axis= 1), np.argmax(fin_targets_ciuo1d, axis= 1)
outputs_ciuo2d, targets_ciuo2d = np.argmax(fin_outputs_ciuo2d, axis= 1), np.argmax(fin_targets_ciuo2d, axis= 1)
BERTciuo1d_df = ciuo1d_X_test.copy()
BERTciuo1d_df['id'] = BERTciuo1d_df.index
BERTciuo1d_df['preds'] = outputs_ciuo1d
BERTciuo1d_df['probs'] = np.max(fin_outputs_ciuo1d, axis=1)
BERTciuo1d_df['preds'] = BERTciuo1d_df['preds'].map(lambda x: labels_unique_ciuo1d[x])
filepath = Path('data_out/ciuo1d/'+'BERTciuo1d'+'.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
BERTciuo1d_df.to_csv(filepath, index=False)


BERTciuo2d_df = ciuo2d_X_test.copy()
BERTciuo2d_df['id'] = BERTciuo2d_df.index
BERTciuo2d_df['preds'] = outputs_ciuo2d
BERTciuo2d_df['probs'] = np.max(fin_outputs_ciuo2d, axis=1)
BERTciuo2d_df['preds'] = BERTciuo2d_df['preds'].map(lambda x: labels_unique_ciuo2d[x])
filepath = Path('data_out/ciuo2d/'+'BERTciuo2d'+'.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
BERTciuo2d_df.to_csv(filepath, index=False)


outputs_caenes1d, targets_caenes1d = np.argmax(fin_outputs_caenes1d, axis= 1), np.argmax(fin_targets_caenes1d, axis= 1)
outputs_caenes2d, targets_caenes2d = np.argmax(fin_outputs_caenes2d, axis= 1), np.argmax(fin_targets_caenes2d, axis= 1)

BERTcaenes1d_df = caenes1d_X_test.copy()
BERTcaenes1d_df['id'] = BERTcaenes1d_df.index
BERTcaenes1d_df['preds'] = outputs_caenes1d
BERTcaenes1d_df['probs'] = np.max(fin_outputs_caenes1d, axis=1)
BERTcaenes1d_df['preds'] = BERTcaenes1d_df['preds'].map(lambda x: labels_unique_caenes1d[x])
filepath = Path('data_out/caenes1d/'+'BERTcaenes1d'+'.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
BERTcaenes1d_df.to_csv(filepath, index=False)

BERTcaenes2d_df = caenes2d_X_test.copy()
BERTcaenes2d_df['id'] = BERTcaenes2d_df.index
BERTcaenes2d_df['preds'] = outputs_caenes2d
BERTcaenes2d_df['probs'] = np.max(fin_outputs_caenes2d, axis=1)
BERTcaenes2d_df['preds'] = BERTcaenes2d_df['preds'].map(lambda x: labels_unique_caenes2d[x])
filepath = Path('data_out/caenes2d/'+'BERTcaenes2d'+'.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
BERTcaenes2d_df.to_csv(filepath, index=False)

print('Predicciones calculadas. Mostrando resultados:')
#compare_models(ciuo1d_y_test, y_pred_ciuo1d, targets_ciuo1d, outputs_ciuo1d, labels_unique_ciuo1d, 'CIUO-1Digito')

#compare_models(ciuo2d_y_test, y_pred_ciuo2d, targets_ciuo1d, outputs_ciuo2d, labels_unique_ciuo2d, 'CIUO-2Digito')

#compare_models(caenes1d_y_test, y_pred_caenes1d, targets_caenes1d, outputs_caenes1d, labels_unique_caenes1d, 'CAENES-1Digito')

#compare_models(caenes2d_y_test, y_pred_caenes2d, targets_caenes2d, outputs_caenes2d, labels_unique_caenes2d, 'CAENES-2Digito')


# El siguiente código genera un archivo csv donde se consolidan las mejores predicciones de cada modelo para cada clasificación
from best_prediction import consolidar_clasificaciones

filepath = Path('data_out/caenes1d/')
consolidar_clasificaciones(filepath)

filepath = Path('data_out/caenes2d/')
consolidar_clasificaciones(filepath)

filepath = Path('data_out/ciuo1d/')
consolidar_clasificaciones(filepath)

filepath = Path('data_out/ciuo2d/')
consolidar_clasificaciones(filepath)
