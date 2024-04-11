import torch

d_embeddings = 50

class DAN(torch.nn.Module):
    def __init__(self):
        super(DAN, self).__init__()
        self.first_layer = torch.nn.Linear(4 * d_embeddings, 512)
        self.dropout = torch.nn.Dropout(0.5)
        self.second_hidden_layer = torch.nn.Linear(512, 512)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.output_layer = torch.nn.Linear(512, 1)

    def forward(self, x):
        return torch.sigmoid(self.output_layer(self.dropout2(torch.relu(self.second_hidden_layer(self.dropout(torch.relu(self.first_layer(x))))))))

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        bidirectional = True
        num_directions = 2 if bidirectional else 1

        self.rnn_sent_one = torch.nn.RNN(input_size=d_embeddings, hidden_size=32, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.rnn_sent_two = torch.nn.RNN(input_size=d_embeddings, hidden_size=32, num_layers=1, batch_first=True, bidirectional=bidirectional)
        self.linear_for_rnns = torch.nn.Linear(32 * 2 * num_directions, 32)
        self.linear_with_word = torch.nn.Linear(32 + d_embeddings, 32)
        self.output_layer = torch.nn.Linear(32, 1)

    def forward(self, x):
        output_1, hidden_1 = self.rnn_sent_one(x[:, :30, :])
        output_2, hidden_2 = self.rnn_sent_two(x[:, 30:60, :])
        combined_hidden = torch.relu(self.linear_for_rnns(torch.cat((output_1[:, -1], output_2[:, -1]), dim=1)))
        combined_with_word = torch.cat((combined_hidden, x[:, 60, :]), dim=1)
        return torch.sigmoid(self.output_layer(torch.relu(self.linear_with_word(combined_with_word))))


class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm_one = torch.nn.LSTM(input_size=d_embeddings, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm_two = torch.nn.LSTM(input_size=d_embeddings, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)
        self.linear_hidden_sent = torch.nn.Linear(64 * 2, 32)
        self.linear_hidden_with_word = torch.nn.Linear(32 + d_embeddings, 32)
        self.batch_norm = torch.nn.BatchNorm1d(32)
        self.output_layer = torch.nn.Linear(32, 1)

    def forward(self, x):
        output_1, (h_n_1, _) = self.lstm_one(x[:, :30, :])
        output_2, (h_n_2, _) = self.lstm_two(x[:, 30:60, :])
        combined_hidden = torch.relu(self.linear_hidden_sent(torch.cat((output_1[:, -1], output_2[:, -1]), dim=1)))
        combined_with_word = torch.cat((combined_hidden, x[:, 60, :]), dim=1)
        return torch.sigmoid(self.output_layer(self.batch_norm(torch.relu(self.linear_hidden_with_word(combined_with_word)))))