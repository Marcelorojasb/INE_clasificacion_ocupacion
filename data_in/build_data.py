# Este archivo crea los set de datos para CIUO-08 y CAENES

import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):

    def __init__(self,dataframe, text, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texto = dataframe[text]
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.texto)

    def __getitem__(self, index):
        texto = str(self.texto[index])
        texto = " ".join(texto.split())

        inputs = self.tokenizer.encode_plus(
            texto,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def build_dataloader(in_df,text,labels):
    '''
    ### build_dataloader(df, text, labels):
    Crea el dataset y el dataloader para llevar a cabo el entrenamiento de la red neuronal. \n
    Un dataLoader, como su nombre indica, permite hacer una carga de los datos, pero añadiendo \n
    características para cargar los datos con dichas características.

    ### Parámetros:
    df: pandas datafrme
        Set de datos a utilizar para el entrenamiento de la red neuronal. 

    text: str, default=None
        Nombre de la columna que contiene los textos a procesar.

    labels: str, default=None
        Nombre de la columna que contiene las etiquetas de los textos a procesar.

    ### Resultados:
    Se obtiene un tupla con los dataloader de la base de entrenamiento y testeo, además de una lista con las etiquetas \n
    únicas presentes en los datos y el dataset de entrenamiento.

    training_loader: DataLoader()
        Dataloader para la base de entrenamiento

    testing_loader: Dataloader()
        Dataloader para la base de prueba

    labels_unique: list
        Lista con etiquetas únicas

    test_dataset: pandas dataframe
        Set de datos de testeo
    '''
    # Obtén la lista de etiquetas únicas
    labels_unique = in_df[labels].unique()
    in_df[text] = in_df[text].astype(str)
    df = in_df[[text,labels]]
    # Crea columnas binarias para cada etiqueta única
    for label in labels_unique:
        # Crea una columna con el nombre de la etiqueta (puedes personalizar el nombre)
        df[f'clase{label}_b'] = (df[labels] == label).astype(int)
    df.drop(columns = [labels], inplace = True)
    df['list'] = df[df.columns[1:]].values.tolist()
    new_df = df[[text, 'list']].copy()
    
    # Defineir variables claves necesarias para el entrenamiento
    MAX_LEN = 200 # Largo máximo de los tokens
    TRAIN_BATCH_SIZE = 32 # Tamaño del batch de entrenamiento
    VALID_BATCH_SIZE = 128 # Tamaño máximo del batch de entrenamiento
    tokenizer = BertTokenizer.from_pretrained('mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es') # Tokenizador entrenado con modelo bert en español
    
    # Crear el dataset y el dataloader para la red neuronal
    train_size = 0.8
    train_dataset=in_df.sample(frac=train_size,random_state=200)
    test_dataset=in_df.drop(train_dataset.index)
    test_dataset = test_dataset.sample(n=100, random_state=42)
    
    X_train = pd.DataFrame(train_dataset[text])
    y_train = train_dataset[labels]
    X_test = pd.DataFrame(test_dataset[text])
    y_test = test_dataset[labels]
    
    train_dataset=new_df.sample(frac=train_size,random_state=200)
    test_dataset=new_df.drop(train_dataset.index)
    test_dataset = test_dataset.sample(n=100, random_state=42).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    
    # Aquí
    
    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset,text, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset,text, tokenizer, MAX_LEN)
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': False,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    return training_loader, testing_loader, labels_unique, test_dataset, X_train, X_test, y_train, y_test
