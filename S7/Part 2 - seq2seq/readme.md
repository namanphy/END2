# Seq 2 Seq Prediction
## Using LSTM (without Attention)

**Datasets**|**Data Preparation**|**Seq2Seq Model ipynb link**|
:-----:|:-----:|:-----:|
[Wikipedia QA Dataset - question & answer pairs](http://www.cs.cmu.edu/~ark/QA-data/)| [readme and source](https://github.com/namanphy/END2/tree/main/S7/Part%202%20-%20seq2seq/data) | [![](https://img.icons8.com/material-rounded/48/000000/github.png)](https://github.com/namanphy/END2/tree/main/S7/Part%202%20-%20seq2seq)
[Quora Dataset - similar question pairs](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)|  [readme and source](https://github.com/namanphy/END2/tree/main/S7/Part%202%20-%20seq2seq/data) | [![](https://img.icons8.com/material-rounded/48/000000/github.png)](https://github.com/namanphy/END2/tree/main/S7/Part%202%20-%20seq2seq)

**The complete information about the preparation of the dataset can be found in above links.** Or click on this [readme](https://github.com/namanphy/END2/tree/main/S7/Part%202%20-%20seq2seq/data).


## 1. QA Dataset

The model is a LSTM ENC-DEC seq2seq prediction. The data preparation. Follwoing is the perplexity loss during training.

![](https://github.com/namanphy/END2/blob/main/S7/Part%202%20-%20seq2seq/imgs/qa_plot.png)


## 2. Quora Dataset

The model is a LSTM ENC-DEC seq2seq prediction. The data preparation. Follwoing is the perplexity loss during training.

![](https://github.com/namanphy/END2/blob/main/S7/Part%202%20-%20seq2seq/imgs/quora_plot.png)

### Problem Encountered
During the inference when the model is given a sequence as source(`src`) tensor and a Zeroes as `trg` tensor(of shape reflecting desired length of output) - The model predicts an output of the shape `[Seq len, Batch size, Embedding dim]`.

Now it is required to convert back this embedding output to its original words to get a readable sequence. This part is showing some difficulties.
