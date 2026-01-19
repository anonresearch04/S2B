import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from utils.preprocessing import pl_bert_preprocessing, sketch_preprocessing
import numpy as np

class PLBertDataset(Dataset):
    def __init__(self, tokenizer, sequences, labels, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq = pl_bert_preprocessing(seq, self.max_length)
        label = self.labels[idx]
        input_ids = [self.tokenizer.convert_tokens_to_ids('[CLS]')] + seq + [self.tokenizer.convert_tokens_to_ids('[SEP]')]
        attention_mask = [1] + [0 if token_id == 0 else 1 for token_id in seq] + [1]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_pl_bert_dataset(tokenizer, df, max_length):
    x = []
    y = []
    for row in df.itertuples(index=False):
        x.append(row.x)
        y.append(row.y)
    pl_bert_dataset = PLBertDataset(tokenizer, x, y, max_length)
    return pl_bert_dataset


def get_pl_bert_dataLoader(datasets, batch_size, shuffle=False):
    return DataLoader(datasets, batch_size=batch_size, shuffle=shuffle)


def get_learned_filter_dataLoader(df, max_length, batch_size=32, shuffle=False):
    input_data_size = len(df)
    input_data_onehot = np.zeros((input_data_size, max_length, 3001), dtype=np.float32)
    labels = []

    for i, row in enumerate(df.itertuples(index=False)):
        data = sketch_preprocessing(row.x, max_length)
        label = row.y
        labels.append(label)
        input_data_onehot[i, np.arange(max_length), np.array(data)] = 1

    labels = np.array(labels)
    X = torch.tensor(input_data_onehot, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, loader
    
    