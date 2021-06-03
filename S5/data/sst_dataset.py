import pytreebank
import os
import pandas as pd
import torch


# Manual Seed
SEED = 45
torch.manual_seed(SEED)

def sentences_to_dataframe(path_sst, split=0.8, seed=SEED):
    assert split > 0.72, "Split ratio is by default 0.72 - Split value must be more then 0.72"

    if path_sst is None:
        path_sst = './stanfordSentimentTreebank'

    dataset = pytreebank.load_sst(path_sst)
    out_path = os.path.join(path_sst, 'sst_{}.txt')

    # Store train, dev and test in separate files
    for category in ['train', 'test', 'dev']:
        with open(out_path.format(category), 'w') as outfile:
            for item in dataset[category]:
                outfile.write("{}\t\t{}\n".format(
                    item.to_labeled_lines()[0][0] + 1,
                    item.to_labeled_lines()[0][1]
                ))

    df_train = pd.read_csv(out_path.format('train'), sep='\t\t', header=None, names=['label', 'text'])
    df_test = pd.concat((pd.read_csv(out_path.format(f), sep='\t\t', header=None, names=['label', 'text']) 
                        for f in ['test', 'dev']), ignore_index=True)

    split_rows = ((df_train.shape[0] + df_test.shape[0]) * split) - df_train.shape[0]
    print(f'Adding {split_rows} more rows to trainset.')
    df_train = pd.concat((df_train, df_test.sample(n=split_rows, random_state=seed)), ignore_index=True)
    df_test = df_test.drop(df_test.sample(n=split_rows, random_state=seed).index)
    df_test.reset_index(inplace=True)
    
    return df_train, df_test
