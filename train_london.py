import time
import torch

import pandas as pd
import numpy as np

from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from math import inf

from source.data_loader import LondonDatasetReader
from source.utils import get_household_complete_data, train_val_test_separator, ToTensor
from source.models import GNNLSTM

#####################################################################
# Defining dataset paths
dataset_paths = {
    'household_info': 'datasets/London/informations_households.csv',
    'acorn_groups': 'datasets/London/acorn_details.csv',
    'weather_daily': 'datasets/London/weather_daily_darksky.csv',
    'weather_hourly': 'datasets/London/weather_hourly_darksky.csv',
    'holidays': 'datasets/London/uk_bank_holidays.csv',
    'daily_block': 'datasets/London/daily_dataset/daily_dataset/',
    'hh_block': 'datasets/London/halfhourly_dataset/halfhourly_dataset/'
}

#####################################################################
# Reading the target half-hourly dataset
print('[*] Generating the dataset...')
available_blocks = [f'block_{i}' for i in range(112)]
target_block = available_blocks[0]

dataframe = pd.read_csv(dataset_paths['hh_block'] + target_block + '.csv')

available_household_ids = list(np.unique(dataframe['LCLid']))

target_households = [available_household_ids[0], available_household_ids[1], available_household_ids[2],
                     available_household_ids[3], available_household_ids[4]]

samples = []
labels = []

for household in target_households:
    x, y = get_household_complete_data(dataframe, household)
    samples.append(x)
    labels.append(y)

train_samples = []
train_labels = []

val_samples = []
val_labels = []

test_samples = []
test_labels = []

print('[*] Generating the Train, Validation, and Test datasets...')
for i in range(len(samples)):
    separated_data = train_val_test_separator(samples[i], labels[i])
    train_samples.append(separated_data[0])
    train_labels.append(separated_data[1])
    val_samples.append(separated_data[2])
    val_labels.append(separated_data[3])
    test_samples.append(separated_data[4])
    test_labels.append(separated_data[5])

#####################################################################
# Generating PyTorch friendly dataset

train_dataset = LondonDatasetReader(train_samples, train_labels, transforms=Compose([ToTensor()]))
val_dataset = LondonDatasetReader(val_samples, val_labels, transforms=Compose([ToTensor()]))
test_dataset = LondonDatasetReader(test_samples, test_labels, transforms=Compose([ToTensor()]))

#####################################################################
# Setting the global parameters
TARGET_NODE = 0
NUM_NODES = len(target_households)
INPUT_DIM = train_samples[0].shape[-1]
OUTPUT_DIM = train_labels[0].shape[-1]
GNN_HIDDEN_SIZES = 64
LSTM_HIDDEN_SIZES = 64
LSTM_NUM_LAYERS = 1
LSTM_DROPOUT = 0.2
BATCH_SIZE = 10
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4

#####################################################################
# Loading the data

print('[*] Generating data loaders...')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#####################################################################
# Building the model, loss function and optimizer

model = GNNLSTM(num_nodes=NUM_NODES,
                input_dim=INPUT_DIM,
                output_dim=OUTPUT_DIM,
                lstm_hidden_size=LSTM_HIDDEN_SIZES,
                lstm_num_layers=LSTM_NUM_LAYERS,
                batch_size=BATCH_SIZE,
                gnn_hidden_size=GNN_HIDDEN_SIZES,
                lstm_dropout=LSTM_DROPOUT,
                target_node=TARGET_NODE)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print('[*] Training started...')
avg_train_losses = []
avg_val_losses = []
avg_test_losses = []

test_preds = []
true_preds = []

best_val_loss = float(inf)

for epoch in range(EPOCHS):
    start_time = time.time()
    avg_train_loss = 0.
    for i, sample in enumerate(train_loader):
        x, y = sample['x'], sample['y']
        if TARGET_NODE is not None:
            y = y[:, TARGET_NODE, :]
        else:
            y = y.sum(dim=1)

        model.graph_lstm.hidden_states = []
        for j in range(NUM_NODES):
            hidden_state = (torch.rand([LSTM_NUM_LAYERS, x.size(0), LSTM_HIDDEN_SIZES]),
                            torch.rand([LSTM_NUM_LAYERS, x.size(0), LSTM_HIDDEN_SIZES]))
            model.graph_lstm.hidden_states.append(hidden_state)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

        avg_train_loss += loss.item() / len(train_dataset)

    avg_train_losses.append(avg_train_loss)
    avg_val_loss = 0.
    avg_test_loss = 0.
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            x, y = sample['x'], sample['y']
            if TARGET_NODE is not None:
                y = y[:, TARGET_NODE, :]
            else:
                y = y.sum(dim=1)

            model.graph_lstm.hidden_states = []
            for j in range(NUM_NODES):
                hidden_state = (torch.rand([LSTM_NUM_LAYERS, x.size(0), LSTM_HIDDEN_SIZES]),
                                torch.rand([LSTM_NUM_LAYERS, x.size(0), LSTM_HIDDEN_SIZES]))
                model.graph_lstm.hidden_states.append(hidden_state)

            out = model(x)
            loss = loss_fn(out, y)

            avg_val_loss += loss.item() / len(val_dataset)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/target_5nodes_k1.pkl')
            print('[*] model saved...')

        avg_val_losses.append(avg_val_loss)
        for i, sample in enumerate(test_loader):
            x, y = sample['x'], sample['y']
            if TARGET_NODE is not None:
                y = y[:, TARGET_NODE, :]
            else:
                y = y.sum(dim=1)

            model.graph_lstm.hidden_states = []
            for j in range(NUM_NODES):
                hidden_state = (torch.rand([LSTM_NUM_LAYERS, x.size(0), LSTM_HIDDEN_SIZES]),
                                torch.rand([LSTM_NUM_LAYERS, x.size(0), LSTM_HIDDEN_SIZES]))
                model.graph_lstm.hidden_states.append(hidden_state)

            out = model(x)
            loss = loss_fn(out, y)

            avg_test_loss += loss.item() / len(test_dataset)

        avg_test_losses.append(avg_test_loss)

    print('[*] Epoch: {}, Avg Train Loss: {:.4f}, Avg Val Loss: {:.4f}, Avg Test Loss: {:.4f}, Elapsed Time: {:.1f}'
          .format(epoch, avg_train_loss, avg_val_loss, avg_test_loss, time.time() - start_time))

np.save('results/avg_train_losses_target_5nodes_k1.npy', np.array(avg_train_losses))
np.save('results/avg_val_losses_target_5nodes_k1.npy', np.array(avg_val_losses))
np.save('results/avg_test_losses_target_5nodes_k1.npy', np.array(avg_test_losses))

# Saving the final test predictions
model.load_state_dict(torch.load('checkpoints/target_5nodes_k1.pkl'))
for i, sample in enumerate(test_loader):
    x, y = sample['x'], sample['y']
    if TARGET_NODE is not None:
        y = y[:, TARGET_NODE, :]
    else:
        y = y.sum(dim=1)

    model.graph_lstm.hidden_states = []
    for j in range(NUM_NODES):
        hidden_state = (torch.rand([LSTM_NUM_LAYERS, x.size(0), LSTM_HIDDEN_SIZES]),
                        torch.rand([LSTM_NUM_LAYERS, x.size(0), LSTM_HIDDEN_SIZES]))
        model.graph_lstm.hidden_states.append(hidden_state)

    out = model(x)

    for j in range(y.size(0)):
        true_preds.append(y[j].tolist())
        test_preds.append(out[j].tolist())

np.save('results/true_preds_target_5nodes_k1.npy', np.array(test_preds))
np.save('results/test_preds_target_5nodes_k1.npy', np.array(test_preds))
