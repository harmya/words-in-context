import torch

d_embeddings = 50

class DAN(torch.nn.Module):
    def __init__(self):
        super(DAN, self).__init__()
        self.first_layer = torch.nn.Linear(4 * d_embeddings, 64)
        self.second_hidden_layer = torch.nn.Linear(64, 32)
        self.output_layer = torch.nn.Linear(32, 1)

    def forward(self, x):
        return torch.sigmoid(self.output_layer(torch.relu(self.second_hidden_layer(torch.relu(self.first_layer(x))))))

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size=d_embeddings, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.linear_1 = torch.nn.Linear(64, 32)
        self.output_layer = torch.nn.Linear(32, 1)


    def forward(self, x):
        _, h_n = self.rnn(x)
        h_n = h_n[-1]
        return torch.sigmoid(self.output_layer(torch.relu(self.linear_1(h_n))))


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=d_embeddings, hidden_size=64, num_layers=2, batch_first=True)
        self.linear_1 = torch.nn.Linear(64, 32)
        self.output_layer = torch.nn.Linear(32, 1)

    def forward(self, x):
        return torch.sigmoid(self.output_layer(torch.relu(self.linear_1(self.lstm(x)[0][:, -1, :]))))
