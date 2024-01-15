import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import unidecode
from spanish_nlp import preprocess



sp = preprocess.SpanishPreprocess(
        lower=True,
        remove_url=True,
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
        remove_vowels_accents=True,
        remove_multiple_spaces=True,
        remove_punctuation=True,
        remove_unprintable=True,
        remove_numbers=True,
        remove_stopwords=True,
        stopwords_list='default',
        lemmatize=False,
        stem=False,
        remove_html_tags=True,
)


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


def predict_single_sample(model, text):
    # Load pre-trained BERT tokenizer
    text = sp.transform(text, debug = False)
    tokenizer = BertTokenizer.from_pretrained('mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es')
    # Tokenize the input text
    tokens = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # Make prediction
    with torch.no_grad():
        model.eval()
        ids = tokens['input_ids'].to(device, dtype=torch.long)
        mask = tokens['attention_mask'].to(device, dtype=torch.long)
        token_type_ids = tokens['token_type_ids'].to(device, dtype=torch.long)
        output = model(ids, mask, token_type_ids)

    # Post-process the prediction
    prediction = torch.sigmoid(output).cpu().detach().numpy().tolist()

    return prediction


def validation(model, testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 1):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets