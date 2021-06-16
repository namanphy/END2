import os
import pandas as pd


# Default Seed
SEED = 45


def sentences_to_dataframe(path_sst : str, split: float=0.8, seed: int=SEED, to_csv: bool=False):

        if seed != 45:
            torch.manual_seed(seed)

        path_dictionary = os.path.join(path_sst + '/dictionary.txt')
        path_phrases = os.path.join(path_sst + '/sentiment_labels.txt')
        path_sentences = os.path.join(path_sst + '/datasetSentences.txt')
        
        dictionary = pd.read_csv(path_dictionary, sep="|", names=['phrase', 'phrase ids'] )
        
        label_mapping = pd.read_csv(path_phrases, sep="|")
        label_mapping['sentiment labels'] = label_mapping['sentiment values'].apply(_discretize_label)

        # sanity check
        assert label_mapping.shape[0] == dictionary.shape[0]
        
        sentences = pd.read_csv(path_sentences, sep="\t", names=['sentence ids', 'sentence'], skiprows=1)
        
        # Merging frames
        sentence_phrase_merge = pd.merge(sentences, dictionary, left_on='sentence', right_on='phrase')
        df = pd.merge(sentence_phrase_merge, label_mapping, on='phrase ids')
        df = df[['sentence', 'sentiment labels']]
        
        # Cleaning sentences
        df['sentence_clean'] = df['sentence'].str.replace(r"\s('s|'d|'re|'ll|'m|'ve|n't)\b", lambda m: m.group(1))
        
        # Saves the complete dataset to csv
        if to_csv:
            df.to_csv("sst_dataset_parsed.csv")
        
        if (split is not None) and (0 < split < 1.0):
            df_train = df.sample(frac=split, random_state=seed)
            df_test = df.loc[~df.index.isin(df_train.index)]

        return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


def _discretize_label(label):
    if label <= 0.2: return 0
    if label <= 0.4: return 1
    if label <= 0.6: return 2
    if label <= 0.8: return 3
    return 4