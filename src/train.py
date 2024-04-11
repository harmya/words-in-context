import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cpu")

import gensim.downloader as api

from neural_archs import DAN, RNN, LSTM
from utils import WiCDataset
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # TODO: change the `default' attribute in the following 3 lines with the choice
    # that achieved the best performance for your case
    parser.add_argument('--neural_arch', choices=['dan', 'rnn', 'lstm'], default='dan', type=str)
    parser.add_argument('--rnn_bidirect', default=False, action='store_true')
    parser.add_argument('--init_word_embs', choices=['scratch', 'glove'], default='scratch', type=str)

    args = parser.parse_args()

    glove_embs = None
    if args.init_word_embs == "glove":
        glove_embs = api.load("glove-wiki-gigaword-50")

    d_embed = glove_embs.vector_size
    print("Embedding size: ", d_embed)
    vocob_size = len(glove_embs.key_to_index)
    print("Vocab size: ", vocob_size)

    def get_X_Y_dataset(model=None, type="train"):
        dataset = WiCDataset(type=type)
        X = None
        Y = torch.tensor(np.array([data["output"] for data in dataset])).reshape(-1, 1).float()

        for i in range(len(dataset)):
            sentence_one = dataset[i]["sentence_one"]
            sentence_two = dataset[i]["sentence_two"]
            sentence_one = np.array(word for word in sentence_one.split())
            sentence_two = np.array(word for word in sentence_two.split())
            word = glove_embs[dataset[i]["word"]]
            word_type = glove_embs[dataset[i]["word_type"]]
            
            if model == "dan":
                X = np.concatenate((sentence_one, sentence_two, word, word_type), axis=0)

            elif model == "rnn" or model == "lstm":
                if len(sentence_one) > 30:
                    print("Sentence one too long")
                    sentence_one = sentence_one[:30]
                else:
                    sentence_one = np.concatenate((sentence_one, np.zeros((30 - len(sentence_one), d_embed))), axis=0)

                if len(sentence_two) > 30:
                    sentence_two = sentence_two[:30]
                else:
                    sentence_two = np.concatenate((sentence_two, np.zeros((30 - len(sentence_two), d_embed))), axis=0)

                X = np.concatenate((sentence_one, sentence_two, word), axis=0)

        dataset = X, Y
        return dataset

    if args.neural_arch == "dan":
        model = DAN().to(torch_device)
    elif args.neural_arch == "rnn":
        if args.rnn_bidirect:
            model = RNN().to(torch_device)
        else:
            model = RNN().to(torch_device)
    elif args.neural_arch == "lstm":
        if args.rnn_bidirect:
            model = LSTM().to(torch_device)
        else:
            model = LSTM().to(torch_device)

    learning_rate = None 
    batch_size = None
    n_epochs = None

    if args.neural_arch == "dan":
        learning_rate = 0.0001
        batch_size = 32
        n_epochs = 200
    elif args.neural_arch == "rnn":
        learning_rate = 0.001
        batch_size = 32
        n_epochs = 60
    elif args.neural_arch == "lstm":
        learning_rate = 0.0005
        batch_size = 32
        n_epochs = 40

    train_dataset = get_X_Y_dataset(model=args.neural_arch)
    print("Train dataset size: ", len(train_dataset[0]))
    print(train_dataset[:10])


    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_loss = []
    val_loss = []

    def validation_loss(model):
        with torch.no_grad():
            dataset = dev_dataset
            X = dataset.tensors[0].to(torch_device)
            Y = dataset.tensors[1].to(torch_device)
            Y_pred = model(X)
            return loss(Y_pred, Y).item()
    
    for epoch in range(n_epochs):
        loss_avg = 0

        for i, (X, Y) in enumerate(dataloader):
            X = X.to(torch_device)
            Y = Y.to(torch_device)
            optimizer.zero_grad()
            Y_pred = model(X)
            loss_val = loss(Y_pred, Y)
            loss_val.backward()
            optimizer.step()
            loss_avg += loss_val.item()

        
        val_loss.append(validation_loss(model))
        training_loss.append(loss_avg / len(dataloader))

    train_accuracy = 0
    test_accuracy = 0
    dev_accuracy = 0

    test_dataset = WiCDataset(type="test")
    test_dataset = get_X_Y_dataset(test_dataset, model=args.neural_arch)

    with torch.no_grad():
        Y_pred_train = model(train_dataset.tensors[0])
        Y_train = train_dataset.tensors[1]
        train_accuracy = sum(torch.round(Y_pred_train) == Y_train)[0] / len(train_dataset)

        Y_pred_test = model(test_dataset.tensors[0])
        Y_test = test_dataset.tensors[1]
        test_accuracy = sum(torch.round(Y_pred_test) == Y_test)[0] / len(test_dataset)

        Y_pred_dev = model(dev_dataset.tensors[0])
        Y_dev = dev_dataset.tensors[1]
        dev_accuracy = sum(torch.round(Y_pred_dev) == Y_dev)[0] / len(dev_dataset)
    
    print("\n------------------------------------------")
    print(f"Neural Architecture: {args.neural_arch}")
    print(f"Train Accuracy: {train_accuracy }")
    print(f"Test Accuracy: {test_accuracy }")
    print(f"Dev Accuracy: {dev_accuracy }")
    print("------------------------------------------\n")

    plt.plot(training_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
