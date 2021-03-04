import time

import pandas as pd
import torch.optim as optim
import torch.nn as nn

from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from source.data_loader import DatasetReader
from source.models import GNNLSTM
from source.utils import ToTensor, series_to_sequence, train_val_test_separator

# Global configurations
train_batch_size = 5
val_batch_size = 5
input_lag = 10
output_lag = 3
train_ratio = 0.8
val_ratio = 0.2
epochs = 10

lstm_hidden_size = 128
lstm_num_layers = 2
learning_rate = 1e-3
weight_decay = 0
#########################
# Prepare dataset
df = pd.read_csv('datasets/AppliancesEnergy/energydata_complete.csv')

time_series_samples, time_series_labels = series_to_sequence(df, 'Appliances', input_lag, output_lag, ['date'])

input_dim = time_series_samples.shape[2]

separated_data = train_val_test_separator(time_series_samples, time_series_labels,
                                          train_ratio=train_ratio, val_ratio=val_ratio)

train_dataset = DatasetReader(separated_data[0], separated_data[1], Compose([ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

val_dataset = DatasetReader(separated_data[2], separated_data[3], Compose([ToTensor()]))
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)

test_dataset = DatasetReader(separated_data[4], separated_data[5], Compose([ToTensor()]))
test_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
########################################################################
# Model, optimizer, and loss definition
model = GNNLSTM(input_dim=input_dim, output_dim=output_lag, lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers, batch_size=train_batch_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    avg_train_loss = 0.
    s_time = time.time()
    for _, sample in enumerate(train_loader):
        x, y = sample['x'], sample['y']

        optimizer.zero_grad()

        out = model(x)

        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

        avg_train_loss += loss.item() / len(train_dataset)

    print('[*] Epoch: {}, Avg Train Loss : {:.4f}, Elapsed Time : {:.2f}'
          .format(epoch, avg_train_loss, time.time() - s_time))




