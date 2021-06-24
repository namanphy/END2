import os
import re
import glob
import subprocess
import pandas as pd


# Default Seed
SEED = 45


def make_qa_dataset():
    
    CHECK_FOLDER = os.path.isdir('../dataset/')
    print(CHECK_FOLDER)
    if not CHECK_FOLDER:
        os.makedirs('../dataset/')
        print("created dataset folder : ", '../dataset/')

    print('Downloading and extracting QA dataset from http://www.cs.cmu.edu/~ark/QA-data/data ..')
    subprocess.call(['wget', 'http://www.cs.cmu.edu/~ark/QA-data/data/Question_Answer_Dataset_v1.2.tar.gz', '-P', '../dataset/'])
    subprocess.call(['tar', '-xf', '../dataset/Question_Answer_Dataset_v1.2.tar.gz', '-C', '../dataset/'])

    dataset = pd.DataFrame()
    
    # get all .txt files from all subdirectories
    all_files = glob.glob('../dataset/Question_Answer_Dataset_v1.2/*/*.txt')

    for file in all_files:
        df = pd.read_csv(file, sep='\t', encoding= 'ISO-8859-1')
        print(f'Complete file path :{file}')
        dataset = pd.concat([dataset, df])
    
    print('\nRemoving Null values ..')
    dataset.dropna(subset=['Question','Answer'], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    print('Final dataset shape : ', dataset.shape)
    
    print('Preprocessing the dataset - lowercase, trim, and remove non-letter characters')
    dataset['Question'] = dataset['Question'].apply(normalizeString)
    
    return dataset


def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s
