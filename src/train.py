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
    
    dataset = WiCDataset(type="train")

    def get_positional_encoding(k, d_embed):
        return torch.tensor([np.sin(k / 10000 ** (2 * i / d_embed)) 
            if i % 2 == 0 else np.cos(k / 10000 ** (2 * i / d_embed)) for i in range(d_embed)])

    def get_X_Y_dataset(dataset, model=None):
        

        X = torch.zeros((len(dataset), d_embed * 4))
        Y = torch.tensor(np.array([data["output"] for data in dataset])).reshape(-1, 1).float()

        for i in range(len(dataset)):
            sentence_one = dataset[i]["sentence_one"]
            sentence_two = dataset[i]["sentence_two"]
            sentence_one = torch.tensor(np.array([glove_embs[word] for word in sentence_one.split() if word in glove_embs]))
            sentence_two = torch.tensor(np.array([glove_embs[word] for word in sentence_two.split() if word in glove_embs]))
            one_idx = dataset[i]["one_index"]
            two_idx = dataset[i]["two_index"]
            
            for j in range(len(sentence_one)):
                sentence_one[j] = sentence_one[j] + get_positional_encoding(j, d_embed)
            
            for j in range(len(sentence_two)):
                sentence_two[j] = sentence_two[j] + get_positional_encoding(j, d_embed)
            
            sentence_one = sentence_one.mean(dim=0)
            sentence_two = sentence_two.mean(dim=0)

            word = torch.tensor(glove_embs[dataset[i]["word"]] if dataset[i]["word"] in glove_embs else np.zeros(d_embed))
            word_type = torch.full((d_embed,), 1 if dataset[i]["word_type"] == "N" else 0)
            word = torch.cat((word, word_type), dim=0)

            input_data = torch.cat((sentence_one, sentence_two, word), dim=0)
            X[i] = input_data

        dataset = torch.utils.data.TensorDataset(X, Y)
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

    train_dataset = get_X_Y_dataset(dataset)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    n_epochs = 400
    
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

        print(f"Epoch: {epoch} Loss: {loss_avg / len(dataloader)}")

    train_accuracy = 0
    test_accuracy = 0
    dev_accuracy = 0

    test_dataset = WiCDataset(type="test")
    test_dataset = get_X_Y_dataset(test_dataset)

    dev_dataset = WiCDataset(type="dev")
    dev_dataset = get_X_Y_dataset(dev_dataset)

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
    
