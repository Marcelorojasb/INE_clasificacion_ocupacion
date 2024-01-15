import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

def calculate_metrics(targets, outputs, average_type, version):
    # Calculate accuracy, F1 scores (micro and macro)
    accuracy = accuracy_score(targets, outputs)
    f1_score_micro = f1_score(targets, outputs, average='micro')
    f1_score_macro = f1_score(targets, outputs, average='macro')
    print(version + ':')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(targets, outputs, average=average_type)

    return precision, recall, f1

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

# Function to compare baseline and BERT models
def compare_models(y_true1, y_baseline, y_true2, y_bert, labels_unique, model_name):
    print(f"Metrics for {model_name} model:")

    # Ensure consistent order of class labels
    sorted_labels = sorted(labels_unique)

    # Calculate metrics for baseline and BERT models
    metrics_baseline = calculate_metrics(y_true1, y_baseline, average_type=None, version='Baseline')
    metrics_bert = calculate_metrics(y_true2, y_bert, average_type=None, version='BERT')

    # Extract precision, recall, and F1 scores
    precision_baseline, recall_baseline, f1_baseline = metrics_baseline
    precision_bert, recall_bert, f1_bert = metrics_bert

    # Create arrays with zeros for missing classes
    y_baseline_extended = np.zeros((len(y_true1), len(sorted_labels)))
    y_bert_extended = np.zeros((len(y_true2), len(sorted_labels)))
    labels_unique = labels_unique.to_list()
    for i, label in enumerate(sorted_labels):
        label_index = labels_unique.index(label)
        y_baseline_extended[:, i] = y_baseline[:, label_index]
        y_bert_extended[:, i] = y_bert[:, label_index]

    # Plot grouped bar graphs for precision, recall, and F1-score
    plot_comparison_bar_graph(precision_baseline, precision_bert, 'Precision', sorted_labels)
    plot_comparison_bar_graph(recall_baseline, recall_bert, 'Recall', sorted_labels)
    plot_comparison_bar_graph(f1_baseline, f1_bert, 'F1 Score', sorted_labels)

# Example usage:
# compare_models(y_true1, y_baseline, y_true2, y_bert, labels_unique, model_name)