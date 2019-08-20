import pandas as pd
import torch
from torch.utils.data import Dataset


class datasets(Dataset):
    def __init__(self, filepath, tokenizer, max_len=64, pad_index=0):
        self.data = pd.read_csv(filepath)
        self.tokenizer = tokenizer
        self.pad_index = pad_index
        self.max_len = max_len
        self.l2i = self.label2ids(self.data['Emotion'])

    def label2ids(self, list):
        unique_label = set(list)
        return {l: i for i, l in enumerate(unique_label)}

    def pad(self, sample):
        diff = self.max_len - len(sample)
        if diff > 0:
            sample = sample + [self.pad_index for _ in range(diff)]
        else:
            sample = sample[:self.max_len]
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tokenize to index
        sentence = self.tokenizer.tokenize(self.data.iloc[idx]['Sentence'], to_ids=True)

        # padding
        sentence = self.pad(sentence)

        # change label str to index
        label = self.l2i[self.data.iloc[idx]['Emotion']]

        return torch.tensor(sentence), torch.tensor(label)

