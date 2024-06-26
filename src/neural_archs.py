import torch

d_embeddings = 50

class DAN(torch.nn.Module):
    def __init__(self, random_embedding=False, vocab_size=0):
        super(DAN, self).__init__()
        self.first_layer = torch.nn.Linear(d_embeddings * 2, 256)
        self.second_hidden_layer = torch.nn.Linear(256, 256)
        self.output_layer = torch.nn.Linear(256, 1)

    def forward(self, x):
        return torch.sigmoid(self.output_layer(torch.relu(self.second_hidden_layer(torch.relu(self.first_layer(x))))))

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        bidirectional = True
        num_directions = 2 if bidirectional else 1

        self.rnn = torch.nn.RNN(input_size=d_embeddings, hidden_size=32, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.linear_for_rnns = torch.nn.Linear(32 * 2 * num_directions, 32)
        self.output_layer = torch.nn.Linear(32, 1)

    def forward(self, x):
        output_1, hidden_1 = self.rnn(x[:, :30, :])
        output_2, hidden_2 = self.rnn(x[:, 30:60, :])
        combined_hidden = torch.relu(self.linear_for_rnns(torch.cat((output_1[:, -1], output_2[:, -1]), dim=1)))
        return torch.sigmoid(self.output_layer(combined_hidden))


class LSTM(torch.nn.Module):
    def __init__(self, bidirectional=False):
        super(LSTM, self).__init__()
        num_directions = 2 if bidirectional else 1
        self.lstm = torch.nn.LSTM(input_size=d_embeddings, hidden_size=64, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.linear_hidden_sent = torch.nn.Linear(64 * 2 * num_directions, 32)
        self.linear_hidden = torch.nn.Linear(32, 32)
        self.output_layer = torch.nn.Linear(32, 1)

    def forward(self, x):
        output_1, (h_n_1, _) = self.lstm(x[:, :30, :])
        output_2, (h_n_2, _) = self.lstm(x[:, 30:60, :])
        combined_hidden = torch.relu(self.linear_hidden_sent(torch.cat((output_1[:, -1], output_2[:, -1]), dim=1)))
        return torch.sigmoid(self.output_layer(torch.relu(self.linear_hidden(combined_hidden))))