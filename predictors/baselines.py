import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#import unidecodeall
#import xgboost as xgb

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from pathlib import Path
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import TweetTokenizer

from datasets import load_dataset, Dataset
from joblib import dump, load

import os


output_folder="data_out"
output_path = os.path.join(os.getcwd(), output_folder)

def predict_baseline(samples,carpeta,versions):
    for version in versions:
        model = load('models/' + version + '.joblib') 
        predicted_probabilities = model.predict_proba(samples)
        max_prob = np.max(predicted_probabilities, axis=1)
        y_pred = model.predict(samples)
        df = samples.copy()
        df['id'] = df.index
        df['preds'] = y_pred
        df['probs'] = max_prob
        #predictions_file_path = os.path.join(output_path)
        os.makedirs('data_out', exist_ok=True)  
        df.to_csv('data_out/'+ carpeta +'/' + version+'.csv')
    return y_pred
    
