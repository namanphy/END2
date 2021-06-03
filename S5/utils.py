import torch
import matplotlib.pyplot as plt

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    _, predictions = torch.max(preds, 1)
    correct = (predictions == y).float() 
    acc = correct.sum() / len(correct)
    return acc



def plot_metric(metrics, legends, xlabel='', ylabel='', title='Plot'):
    
    if type(metrics != list):
        metrics = [metrics]
    if type(legends != list):
        legends = [legends]
    assert len(metrics) < 4, "Too many metrics. only 3 are supported."

    marker = ['o', 'x', 'd']
    i = 0
    for metric in metrics:
        plt.plot(range(len(metric)), metric, marker=marker[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legends)
        plt.title(title)
        plt.show()
