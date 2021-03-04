import torch

import torch.nn as nn
import numpy as np

from torch_geometric.nn import DenseGCNConv
from sklearn.neighbors import kneighbors_graph
from source.layers import GraphLSTM


class GNNLSTM(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, lstm_hidden_size,
                 lstm_num_layers, batch_size, gnn_hidden_size, lstm_dropout=0, **kwargs):
        super(GNNLSTM, self).__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.gnn_hidden_size = gnn_hidden_size
        self.lstm_dropout = lstm_dropout
        self.output_dim = output_dim
        self.batch_size = batch_size

        if 'target_node' in kwargs.keys():
            self.target_node = kwargs['target_node']
        else:
            self.target_node = None

        # LSTM layers definition
        self.graph_lstm = GraphLSTM(num_nodes=num_nodes,
                                    input_dim=input_dim,
                                    hidden_size=lstm_hidden_size,
                                    num_layers=lstm_num_layers,
                                    batch_size=batch_size,
                                    dropout=lstm_dropout)

        # GNN layers definition
        self.gcn = DenseGCNConv(in_channels=lstm_num_layers * lstm_hidden_size, out_channels=gnn_hidden_size)

        # MLP definition
        self.mlp = nn.Sequential(
            nn.Linear(gnn_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        lstm_outputs, lstm_hidden_states = self.graph_lstm(x)

        lstm_feature_vectors = torch.zeros([x.size(0), self.num_nodes,
                                            self.lstm_num_layers * self.lstm_hidden_size], dtype=torch.float32)

        for i in range(self.num_nodes):
            size = lstm_hidden_states[i][0].size()
            try:
                lstm_feature_vectors[:, i, :] = lstm_hidden_states[i][0].view(size[1], size[0] * size[2])
            except:
                print(size[1], size[0] * size[2])
                print(lstm_feature_vectors[:, i, :].size())

        adj = self.generate_adjacency_matrix(lstm_feature_vectors.detach().numpy())

        out = torch.relu(self.gcn(lstm_feature_vectors, adj))

        if self.target_node is not None:
            read_out = out[:, self.target_node, :]
        else:
            read_out = out.mean(dim=1)

        out = self.mlp(read_out)

        return out

    @staticmethod
    def generate_adjacency_matrix(feature_vectors, mode='knn'):
        covariances = torch.zeros([feature_vectors.shape[0], feature_vectors.shape[1], feature_vectors.shape[1]])

        if mode == 'cov':
            for batch in range(feature_vectors.shape[0]):
                cov = np.cov(feature_vectors[batch])
                covariances[batch] = torch.tensor(cov)

            covariances[covariances >= 0.5] = 1.
            covariances[covariances < 0.5] = 0.
        else:
            for batch in range(feature_vectors.shape[0]):
                matrix = kneighbors_graph(feature_vectors[batch], n_neighbors=1).toarray()
                covariances[batch] = torch.tensor(np.clip(matrix + matrix.T, a_min=0, a_max=1))

        # np.save('sample_graphs.npy', covariances)
        # exit()

        return covariances
