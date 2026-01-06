import numpy as np
import pandas as pd
import re
import csv
import preprocessor as p
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import *
import math

from utils.author_100recent import get_100_recent_posts
import utils.dataset_processors as dataset_processors


def load_raw_text_dataset(dataset, tokenizer, token_length, mode):
    if dataset == "essays":
        datafile = "data/essays/essays.csv"
        author_ids, input_ids, targets = dataset_processors.essays_embeddings(
            datafile, tokenizer, token_length, mode
        )
    elif dataset == "kaggle":
        datafile = "data/kaggle/kaggle.csv"
        author_ids, input_ids, targets = dataset_processors.kaggle_embeddings(
            datafile, tokenizer, token_length
        )
    elif dataset == "pandora":
        author_ids, input_ids, targets = dataset_processors.pandora_embeddings(
            datafile, tokenizer, token_length
        )
    return author_ids, input_ids, targets


class MyMapDataset(Dataset):
    def __init__(self, dataset, tokenizer, token_length, DEVICE, mode):
        author_ids, input_ids, targets = load_raw_text_dataset(dataset, tokenizer, token_length, mode)
        author_ids = torch.from_numpy(np.array(author_ids)).long().to(DEVICE)
        # input_ids = torch.from_numpy(np.array(input_ids)).long().to(DEVICE)
        from torch.nn.utils.rnn import pad_sequence
        input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=0
        ).to(DEVICE)
        targets = torch.from_numpy(np.array(targets))

        if dataset == "pandora":
            targets = targets.float().to(DEVICE)
        else:
            targets = targets.long().to(DEVICE)

        self.author_ids = author_ids
        self.input_ids = input_ids
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.author_ids[idx], self.input_ids[idx], self.targets[idx])
