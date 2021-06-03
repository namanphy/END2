# Sentiment Prediction
### On Standford Sentiment Treebank using LSTM

--------
# Hyperparameters
- Batch size: 64
- Learning rate: 0.0001
- Epochs: 100
- Optimizer: Adam
- Criterion: cross entropy loss
 
# Knowing the Data
SST is a popular dataset used for benchmarking NLP related models. It comprises of movies reviews and sentiments associated with reviews manually labeled for nearly 25 classes. 
Here we have used a total five labels ["very negative", "negative", "neutral", "positive", "very positive"].

The Dataset has been divided into train, test split of 80-20 %.

1. On exploring the dataset, it was found to be imbalanced dataset. It is clear that most of the training samples belong to classes 2 and 4 (the weakly negative/positive classes)

![i]()

2. The length of the texts in the dataset is also skewed with mean Length --.

![i]()


# Data Augmentation - [notebook](https://github.com/namanphy/END2/blob/main/S5/Data_Augmentation.ipynb)
**The size of training data increased from 9484 to 14700 datapoints using the augmentations.**

From the reference of the [paper]() the follwoing augmentation techniques were applied to the training dataset.

![i]()

1. Random swap: The random swap augmentation takes a sentence and then swaps words randomly within it n times.

2. Random delete: The random delete augmentationn randomly deletes words from a sentence for a given a probability.

3. Back Translate: This method invokes the translation of the text to random language and then translate it back to the required language using pretrained language models. Here, Google Translate is used to do the needful.

*Google Translate has api limits - and thus only 5% of the training dataset could be used for back translation.*

The augmented data is saved in three different csv files which are present in the repository.

### vocab size
![i]()

# Model - Bi-directional LSTM

The network we decided to build is designed as follows:

- Embedding Layer: Cconverts our word tokens (integers) into embedding of specific size.
- LSTM Layer: Defined by hidden state dims and number of layers.
- Fully Connected Layer: Maps output of LSTM layer to a desired output size.
- Output: Softmax output from the last timestep is considered as the final output of this network.

![i]()

| | |
|---|---|
|Input dim | vocab size (~19K)| 
|Embeddings dim | 100 |
| Bidirectional LSTM | 2 layers |
| Bidirectional LSTM - Hidden dim | 256 |
| Dropout | 0.4 |
| Fully Connected layer - dim | 256 |
| Output dim | 5 |


# Results
### ACCURACY ACHIEVED - 40.78%

The model was overfitting in most of the scenarios. 

# Evaluation - What's happened?

Following are the examples from the test set - when the trained model is used to predict the label. Some 

