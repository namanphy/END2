# Sentiment Prediction
### On Standford Sentiment Treebank using LSTM

--------

## Solution notebook - [Stanford  Sentiment Treebank - Classification LSTM](link)

# Hyperparameters
- Batch size:
- Learning rate:
- Epochs:
- Optimizer: Adam
- Criterion: cross entropy loss
 
# Knowing the Data
SST is a popular dataset used for benchmarking NLP related models. It comprises of movies reviews and sentiments associated with reviews manually labeled for nearly 25 classes. 
Here we have used a total five labels ["very negative", "negative", "neutral", "positive", "very positive"].

The Dataset has been divided into train, test split of 80-20 %.

1. On exploring the dataset, it was found to be imbalanced dataset. It is clear that most of the training samples belong to classes 2 and 4 (the weakly negative/positive classes)

![i](https://github.com/namanphy/END2/blob/main/S7/Part%201%20-%20sentiment%20classification/imgs/sample-dist.png)

2. The length of the texts in the dataset is also skewed with mean Length --.

![i](https://github.com/namanphy/END2/blob/main/S7/Part%201%20-%20sentiment%20classification/imgs/length-dist.png)


**The data is divided into multiple txt files. These files are combined to generate a dataset.** After preprocessing, the
data is saved in new csv file [sst_dataset_parsed.csv](https://github.com/namanphy/END2/blob/main/S7/Part%201%20-%20sentiment%20classification/data/sst_dataset_parsed.csv). There is a preprocessing notebook as well as a 
python file.

## Data Preparation source code - [py file](https://github.com/namanphy/END2/tree/main/S7/Part%201%20-%20sentiment%20classification/data/sst_dataset.py)

--------

# Model - Bi-directional LSTM

The network we decided to build is designed as follows:

- Embedding Layer: Cconverts our word tokens (integers) into embedding of specific size.
- LSTM Layer: Defined by hidden state dims and number of layers.
- Fully Connected Layer: Maps output of LSTM layer to a desired output size.
- Output: Softmax output from the last timestep is considered as the final output of this network.

![i](https://github.com/namanphy/END2/blob/main/S5/imgs/chrome_ImJOo4siM2.png)

| | |
|---|---|
|Input dim | vocab size (~19K)| 
|Embeddings dim | 100 |
| Bidirectional LSTM | 2 layers |
| Bidirectional LSTM - Hidden dim | 256 |
| Dropout | 0.4 |
| Fully Connected layer - dim | 256 |
| Output dim | 5 |


# Results & Experiments
### ACCURACY ACHIEVED - 40.78%

The model was overfitting in all the scenarios. The accuracy and loss was measured for base line
model, the gradually augmentations were added, and model architectures/hyperparams were experimented and measured for accuracy.

The best case scenario is following. **The loss was clearly reached its lowest at 15-20 epochs and after that model only overfitted.**

### Loss
![](https://github.com/namanphy/END2/blob/main/S7/Part%201%20-%20sentiment%20classification/imgs/loss.png)

### Accuracy
![](https://github.com/namanphy/END2/blob/main/S7/Part%201%20-%20sentiment%20classification/imgs/acc.png)

# Evaluation - What's happened?

Following confusion matrix is plotted on the results of the test data. It shows a better picture of the model than the 
accuracy metric.

![](https://github.com/namanphy/END2/blob/main/S7/Part%201%20-%20sentiment%20classification/imgs/confusion_matrix.png)
