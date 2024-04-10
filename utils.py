from torch.utils.data import Dataset
import pandas as pd
import os
import re

class WiCDataset(Dataset):
    def __init__(self):
        self.data = []
        train_data_path = "train/train.data.txt"
        train_data = None

        with open(train_data_path, "r") as f:
            train_data = f.read()

        for data_point in train_data.split("\n"):
            attributes = data_point.split("\t")
            if len(attributes) == 5:
                word = attributes[0]
                word_type = attributes[1]
                sent_one_index = int(attributes[2].split("-")[0])
                sent_two_index = int(attributes[2].split("-")[1])
                sentence_one = self.preprocess(attributes[3])
                sentence_two = self.preprocess(attributes[4])
                self.data.append({
                    "word": word,
                    "word_type": word_type,
                    "one_index": sent_one_index,
                    "two_index": sent_two_index,
                    "sentence_one": sentence_one,
                    "sentence_two": sentence_two
                })
                

    def preprocess(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z]+", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence.strip()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

