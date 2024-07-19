# Este archivo permite obtener métricas y comparaciones de las predicciones realizadas por los modelos entrenados

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

def calculate_metrics(targets, outputs, average_type, version):
    # Calcular accuracy, F1 scores (micro and macro)
    accuracy = accuracy_score(targets, outputs)
    f1_score_micro = f1_score(targets, outputs, average='micro')
    f1_score_macro = f1_score(targets, outputs, average='macro')
    print(version + ':')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    # Calcular precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(targets, outputs, average=average_type)

    return precision, recall, f1
    
# Función para crear gráficos de comparación entre modelos
def plot_comparison_bar_graph(metric_values_baseline, metric_values_bert, metric_name, class_labels):
    bar_width = 0.35
    index = np.arange(len(class_labels))

    fig, ax = plt.subplots(figsize=(15, 8))
    baseline_bars = ax.bar(index, metric_values_baseline, bar_width, label='Baseline')
    bert_bars = ax.bar(index + bar_width, metric_values_bert, bar_width, label='BERT')

    ax.set_xlabel('Class')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison per Class')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(class_labels)
    ax.legend()

    plt.show()

# Función para comparar modelos baseline con modelo BERT
def compare_models(y_true1, y_baseline, y_true2, y_bert, labels_unique, model_name):
    '''
    ### compare_models(y_true1, y_baseline, y_true2, y_bert, labels_unique, model_name):
    Entrega gráficos comparativos de las métricas de los diferentes modelos evaluados

    ### Parámetros:
    y_true1: list
        Valores de etiquetas de clases reales
    
    y_baseline: list
        Valores de etiquetas predichas por modelo baseline

    y_bert: list
        Valores de etiquetas predichas por modelo BERT

    labels_unique: list
        Listado de etiquetas a predecir

    model_name: str, {ciuo_1d, ciuo_2d, caenes_1d, caenes_2d}
    '''
    print(f"Metrics for {model_name} model:")

    # Asegurarse de un orden consistente en las etiquetas de clase
    sorted_labels = sorted(labels_unique)

    # Calcular las métricas para los baselines y BERT
    metrics_baseline = calculate_metrics(y_true1, y_baseline, average_type=None, version='Baseline')
    metrics_bert = calculate_metrics(y_true2, y_bert, average_type=None, version='BERT')

    # Extraer precision, recall, and F1 scores
    precision_baseline, recall_baseline, f1_baseline = metrics_baseline
    precision_bert, recall_bert, f1_bert = metrics_bert

    # Crear arreglos con ceros para las clases faltantes
    y_baseline_extended = np.zeros((len(y_true1), len(sorted_labels)))
    y_bert_extended = np.zeros((len(y_true2), len(sorted_labels)))
    labels_unique = labels_unique.to_list()
    for i, label in enumerate(sorted_labels):
        label_index = labels_unique.index(label)
        y_baseline_extended[:, i] = y_baseline[:, label_index]
        y_bert_extended[:, i] = y_bert[:, label_index]

    # Graficar gráficos de barra agrupados para precision, recall y F1-score
    plot_comparison_bar_graph(precision_baseline, precision_bert, 'Precision', sorted_labels)
    plot_comparison_bar_graph(recall_baseline, recall_bert, 'Recall', sorted_labels)
    plot_comparison_bar_graph(f1_baseline, f1_bert, 'F1 Score', sorted_labels)

# Ejempo de uso:
# compare_models(y_true1, y_baseline, y_true2, y_bert, labels_unique, 'ciuo_1d_24-08-15')
