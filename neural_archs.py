import torch

d_embeddings = 50

class DAN(torch.nn.Module):
    def __init__(self):
        super(DAN, self).__init__()
        self.first_hidden_layer = torch.nn.Linear(3 * d_embeddings, 64)
        self.second_hidden_layer = torch.nn.Linear(64, 64)
        self.output_layer = torch.nn.Linear(64, 1)

    def forward(self, x):
        return self.output_layer(torch.relu(self.second_hidden_layer(torch.relu(self.first_hidden_layer(x)))))


class RNN(torch.nn.Module):
    def __init__(self):
        # TODO: Declare RNN model architecture
        super(RNN, self).__init__()


    def forward(self, x):
        # TODO: Implement RNN forward pass
        pass


class LSTM(torch.nn.Module):
    def __init__(self):
        # TODO: Declare LSTM model architecture
        super(LSTM, self).__init__()


    def forward(self, x):
        # TODO: Implement LSTM forward pass
        pass
