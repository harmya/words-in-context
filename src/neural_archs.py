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
        self.rnn_sent_one = torch.nn.RNN(input_size=d_embeddings, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.rnn_sent_two = torch.nn.RNN(input_size=d_embeddings, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.linear_for_rnns = torch.nn.Linear(64 * 2, 32)
        self.linear_with_word = torch.nn.Linear(32 + d_embeddings, 32)
        self.output_layer = torch.nn.Linear(32, 1)


    def forward(self, x):
        _, hidden_1 = self.rnn_sent_one(x[:, :30, :])
        _, hidden_2 = self.rnn_sent_two(x[:, 30:60, :])
        h_n_1 = hidden_1[-1]
        h_n_2 = hidden_2[-1]
        combined_hidden = torch.relu(self.linear_for_rnns(torch.cat((h_n_1, h_n_2), dim=1)))
        combined_with_word = torch.cat((combined_hidden, x[:, 60, :]), dim=1)
        return torch.sigmoid(self.output_layer(torch.relu(self.linear_with_word(combined_with_word))))


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=d_embeddings, hidden_size=64, num_layers=2, batch_first=True)
        self.linear_1 = torch.nn.Linear(64, 32)
        self.output_layer = torch.nn.Linear(32, 1)

    def forward(self, x):
        return torch.sigmoid(self.output_layer(torch.relu(self.linear_1(self.lstm(x)[0][:, -1, :]))))
