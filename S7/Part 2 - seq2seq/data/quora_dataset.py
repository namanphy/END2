import os
import re
import glob
import subprocess
import pandas as pd


# Default Seed
SEED = 45


def make_quora_dataset():
    
    CHECK_FOLDER = os.path.isdir('../dataset/')
    print(CHECK_FOLDER)
    if not CHECK_FOLDER:
        os.makedirs('../dataset/')
        print("created dataset folder : ", '../dataset/')

    print('Downloading Quora dataset from http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv ..')
    subprocess.call(['wget', 'http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv', '-P', '../dataset/'])

    dataset = pd.read_csv("../dataset/quora_duplicate_questions.tsv", sep='\t')
    dataset = dataset[dataset['is_duplicate']==1]
    dataset.reset_index(drop=True, inplace=True)
    print('Final dataset shape : ', dataset.shape)
    
    print('Preprocessing the dataset - lowercase, trim, and remove non-letter characters')
    dataset['question1'] = dataset['question1'].apply(normalizeString)
    dataset['question2'] = dataset['question2'].apply(normalizeString)
    print('Done.')
    return dataset


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s
