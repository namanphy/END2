# Data Preparation

There are two Sequence based datasets. The link to .py files of preprocessing to both datasets is given.

**Datasets**|**Preprocess Source**|
:-----:|:-----:|
[Wikipedia QA Dataset - question & answer pairs](http://www.cs.cmu.edu/~ark/QA-data/)|[source](https://github.com/namanphy/END2/blob/main/S7/Part%202%20-%20seq2seq/data/qa_dataset.py)
[Quora Dataset - similar question pairs](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)|[source](https://github.com/namanphy/END2/blob/main/S7/Part%202%20-%20seq2seq/data/quora_dataset.py)


## 1. Wikepedia QA Dataset

**Dataset Info** - This dataset includes Wikipedia articles, factual questions manually generated from them, and answers to these manually generated questions for use in academic research.

The file "question_answer_pairs.txt" contains the questions and answers. The first line of the file contains 
column names for the tab-separated data fields in the file. This first line follows:

1. Field 1 - **ArticleTitle** - It is the name of the Wikipedia article from which questions and answers initially came.
2. Field 2 - **Question** - It is the question.
3. Field 3 - **Answer** - It is the answer.
4. Field 4 - **DifficultyFromQuestioner** - It is the prescribed difficulty rating for the question as given to the question-writer. 
5. Field 5 - **DifficultyFromAnswerer** - It is a difficulty rating assigned by the individual who evaluated and answered the question, which may differ from the difficulty in field 4.
6. Field 6 - **ArticleFile** - It is the relative path to the prefix of the article files. html files (.htm) and cleaned 
text (.txt) files are provided.


## 2. Quora Dataset
 **Dataset Info** - This dataset of Quora questions to determine whether pairs of question texts actually correspond to semantically equivalent queries. In essence, the data released by quora conatins set of questions with alternative ways of asking the same questions.

The dataset consists of over **400,000 lines of potential question duplicate pairs**.The dataset is available in TSV file. It is downloaded and processed as pandas dataframe. 

There are 6 columns in the dataset of which `question1` and `question2` are thw two sequnces of interest. The `is_duplicate` field determine whether the question is duplicated.

**Only rows with `is_duplicate=1` are considered here.**


# Process
For both the datasets a similar is process is followed for preprocessing.

1. The dataset is checked for null values and duplicate values. If found they were removed from the dataset.

2. The resulting dataset text is then normalized i.e. converted to lowercase, and trimmed for non-letter characters.

```
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s
```
3. Finally, using torchtext's legacy api - the vocab is generated form the data and dataloaders are created.


## Reference

Both the datasets are open source and are picked from :

1. http://www.cs.cmu.edu/~ark/QA-data/ 

2. https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs
