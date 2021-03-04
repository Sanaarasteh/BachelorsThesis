import torch

import torch.nn as nn


class GraphLSTM(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_size, num_layers, batch_size, dropout=0):
        super(GraphLSTM, self).__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout

        self.lstm_modules = nn.ModuleList()

        for i in range(self.num_nodes):
            if num_layers > 1:
                lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                               batch_first=True, dropout=self.dropout)
            else:
                lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
                               batch_first=True)

            self.lstm_modules.append(lstm)

        # initialize hidden and cell states of the LSTM modules
        self.hidden_states = []

        for i in range(self.num_nodes):
            hidden_state = (torch.rand([self.num_layers, self.batch_size, self.hidden_size]),
                            torch.rand([self.num_layers, self.batch_size, self.hidden_size]))
            self.hidden_states.append(hidden_state)

    def forward(self, x):
        # Note: the shape of x is (Batch, #Nodes, #Sequence, Dimension)
        outputs = torch.zeros([self.num_nodes, x.size(0), x.size(2), self.hidden_size], dtype=torch.float32)

        for i in range(self.num_nodes):
            output, self.hidden_states[i] = self.lstm_modules[i](x[:, i, :, :], self.hidden_states[i])
            outputs[i] = output

        return outputs, self.hidden_states
