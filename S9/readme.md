# Assignment 9

# Objective
Implement the following metrics (either on separate models or same, your choice):
* Recall, Precision, and F1 Score
* BLEU 
* Perplexity (explain whether you are using bigram, trigram, or something else, what does your PPL score represent?)
* BERTScore (here are [1](https://colab.research.google.com/drive/1kpL8Y_AnUUiCxFjhxSrxCsc6-sDMNb_Q), [2](https://huggingface.co/metrics/bertscore) examples)

Once done, proceed to answer questions in the Assignment-Submission Page. 
Questions asked are:
* Share the link to the readme file where you have explained all 4 metrics. 
* Share the link(s) where we can find the code and training logs for all of your 4 metrics
* Share the last 2-3 epochs/stage logs for all of your 4 metrics separately (A, B, C, D) and describe your understanding about the numbers you're seeing, are they good/bad? Why?


## Solution

**Names**|**Files ipynb link**|
:-----:|:-----:|
Precision - Recall - F1 score | [![](https://img.icons8.com/material-rounded/48/000000/github.png)]()
BLEU - BertScore - PPL | [![](https://img.icons8.com/material-rounded/48/000000/github.png)]()


# Evaluation Metrics

Evalauation metrics quantifies the performance of a machine learning model. 'Accuracy' is one of the most commonly used Evaluation metric in the classification and regression problems, that is the total number of correct predictions by the total number of predictions.

## Classification Metrics
Accuracy, Recall, Precision, and F1-Score, ROC-AUC Curve are the key classification metrics.


### Accuracy:
It is defined as percentage of total number of correct predictions to the total number of observations in the dataset. It can be easily calculated as total numer of correct predictions divided by total number of predictions.

### Recall:
Recall is the percentage of relevant results that are correctly classified by the model, ie,it is the ratio of samples which were predicted to belong to a class with respect to all of the samples that truly belong in the class (predicted results).

### Precision
Precision is the percentage of relevant results, ie, it is the ratio of True Positives(TP) to all the positives in the dataset (actual results).

### F1 Score
F1 is the weighted average of precision and recall of the model. It gives more importance to the false positives and false negatives while not letting large numbers of true negatives influence the score. A good F1 score is when there are low false positives and low false negatives. The F1 score ranges from 0 to 1 and is considered perfect when it's 1.


## Sequence Prediction Metrics

### BLEU Score
The BLEU (BiLingual Evaluation Understudy) score is a string-matching algorithm used for evaluating the quality of text which has been translated by a model from a language. The bleu metric ranges from 0 to 1 with 0 being the lowest score and 1 the highest. It signifies what percentage of n-grams can be found in the prediction/translation.


### Perplexity
 Perplexity (PPL) is one of the most common metrics for evaluating LM. Perplexity is defined as a measurement of how well a probability distribution or probability model predicts a sample. A better language model will have lower perplexity values or higher probability values for a test/valid set. 
 It is defined as exponential average negative log-likelihood of a sequence.


### BERTScore
 BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference sentences by cosine similarity. BertScore computes a similarity score and generates scores in three common metrics- precision, recall and F1 measure.


 ### Group Members
 1. Rangasai
 2. Naman Bhardwaj
