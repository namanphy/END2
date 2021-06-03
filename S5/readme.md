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

![i](https://github.com/namanphy/END2/blob/main/S5/imgs/distribution-lables.png)

2. The length of the texts in the dataset is also skewed with mean Length --.

![i](https://github.com/namanphy/END2/blob/main/S5/imgs/distribution-lengths.png)


# Data Augmentation - [notebook](https://github.com/namanphy/END2/blob/main/S5/Data_Augmentation.ipynb)
**The size of training data increased from 9484 to 14700 datapoints using the augmentations.**

From the reference of the [paper]() the follwoing augmentation techniques were applied to the training dataset.

![i](https://github.com/namanphy/END2/blob/main/S5/imgs/chrome_ICcYdhQbYM.png)

1. Random swap: The random swap augmentation takes a sentence and then swaps words randomly within it n times.

    ```
    aug = naw.RandomWordAug(action="swap", aug_p=20, aug_min=2, aug_max=10)
    augmented_text = aug.augment(text)
    ```
2. Random delete: The random delete augmentationn randomly deletes words from a sentence for a given a probability.

    ```
    aug = naw.RandomWordAug(action="delete", aug_p=20, aug_min=2, aug_max=10)
    augmented_text = aug.augment(text)
    ```
3. Back Translate: This method invokes the translation of the text to random language and then translate it back to the required language using pretrained language models. Here, Google Translate is used to do the needful.

*Google Translate has api limits - and thus only 5% of the training dataset could be used for back translation.*

The augmented data is saved in three different csv files which are present in the repository.

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
![](https://github.com/namanphy/END2/blob/main/S5/imgs/loss.png)

### Accuracy
![](https://github.com/namanphy/END2/blob/main/S5/imgs/accuracy.png)

# Evaluation - What's happened?

Following are the examples from the test set - when the trained model is used to predict the label.

![](https://github.com/namanphy/END2/blob/main/S5/imgs/chrome_jndK7PXAyt.png)

# Learning
There is a dire need to incorporate a experiments tracking tool or library in my workflow. Lots of experiments but a very sparsely documented.
