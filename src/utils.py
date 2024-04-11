from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

class WiCDataset(Dataset):
    def __init__(self, type="train"):
        self.data = []
        data_path = None
        data_output_path = None
        self.word_to_index = {}

        if type == "train":
            data_path = "../train/train.data.txt"
            data_output_path = "../train/train.gold.txt"
        elif type == "test":
            data_path = "../test/test.data.txt"
            data_output_path = "../test/test.gold.txt"
        elif type == "dev":
            data_path = "../dev/dev.data.txt"
            data_output_path = "../dev/dev.gold.txt"

        with open(data_output_path, "r") as f:
            train_data_output = f.read()

        train_data_output = np.array([1 if x == "T" else 0 for x in train_data_output.split("\n")])

        with open(data_path, "r") as f:
            train_data = f.read()

        for data_point, output in zip(train_data.split("\n"), train_data_output):
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
                    "sentence_two": sentence_two,
                    "output": output
                })

                if word not in self.word_to_index:
                    self.word_to_index[word] = len(self.word_to_index)
                for word in sentence_one.split():
                    if word not in self.word_to_index:
                        self.word_to_index[word] = len(self.word_to_index)
                for word in sentence_two.split():
                    if word not in self.word_to_index:
                        self.word_to_index[word] = len(self.word_to_index)
        

    def preprocess(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z]+", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence)
        sentence = " ".join([word for word in sentence.split() if word not in stop_words])
        if len(sentence.split()) == 0:
            sentence = "empty"
        return sentence.strip()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
