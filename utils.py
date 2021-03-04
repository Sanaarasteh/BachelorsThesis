import torch
import random

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

pd.options.mode.chained_assignment = None


def series_to_sequence(dataset, target_column, input_lag, output_lag, remove_columns=None):
    x_columns = list(dataset.columns)
    x_columns.remove(target_column)

    if remove_columns is not None:
        for column in remove_columns:
            x_columns.remove(column)

    x = dataset[x_columns].values
    y = dataset[target_column].values

    ds_length = dataset.shape[0]

    new_samples = []
    new_labels = []

    for i in range(0, ds_length - (input_lag + output_lag)):
        new_sample = x[i: i + input_lag]
        new_label = y[i + input_lag: i + input_lag + output_lag]

        new_samples.append(new_sample)
        new_labels.append(new_label)

    return np.array(new_samples, dtype=np.float), np.array(new_labels, np.float)


def train_val_test_separator(samples, labels, train_ratio=0.8, val_ratio=0.2):
    ds_length = samples.shape[0]
    indices = [i for i in range(ds_length)]

    train_val_length = int(train_ratio * ds_length)

    train_val_indices = random.choices(indices, k=train_val_length)
    test_indices = sorted(list(set(indices).difference(set(train_val_indices))))

    val_indices = random.choices(train_val_indices, k=int(val_ratio * len(train_val_indices)))
    train_indices = sorted(list(set(train_val_indices).difference(set(val_indices))))

    train_samples, train_labels = samples[train_indices], labels[train_indices]
    val_samples, val_labels = samples[val_indices], labels[val_indices]
    test_samples, test_labels = samples[test_indices], labels[test_indices]

    return train_samples, train_labels, val_samples, val_labels, test_samples, test_labels


def cleaned_household_data(dataset, household_id):
    data = dataset[dataset['LCLid'] == household_id]

    low_anomalies = get_low_anomalies(data)
    dataset_chunks = []

    new_data = data.values
    for i in range(len(low_anomalies)):
        index = low_anomalies[i]['index']
        previous_date = low_anomalies[i]['previous_date']
        new_date = timedelta(minutes=30) + previous_date
        new_date = datetime.strftime(new_date, '%Y-%m-%d %H:%M:%S.%f')
        energy = low_anomalies[i]['energy']
        new_data[index] = np.array([household_id, new_date, energy])
        new_data = np.delete(new_data, index + 1, 0)

    data = pd.DataFrame(new_data, columns=list(data.columns))

    high_anomalies = get_high_anomalies(data)

    dummies = time_to_one_hot(data)

    dummies['energy(kWh/hh)'] = data['energy(kWh/hh)']

    for i in range(len(high_anomalies)):
        if i == 0:
            dataset_chunks.append(dummies[: high_anomalies[i]['index']])
        elif i == len(high_anomalies) - 1:
            dataset_chunks.append(dummies[high_anomalies[i - 1]['index']: high_anomalies[i]['index']])
            dataset_chunks.append(dummies[high_anomalies[i]['index']:])
        else:
            dataset_chunks.append(dummies[high_anomalies[i - 1]['index']: high_anomalies[i]['index']])

    return dataset_chunks


def get_household_complete_data(dataset, household):
    chunks = cleaned_household_data(dataset, household)

    x_values = []
    y_values = []

    for chunk in chunks:
        if chunk.shape[0] >= 8:
            series = series_to_sequence(chunk, 'energy(kWh/hh)', 6, 2)

            x_values.append(series[0])
            y_values.append(series[1])

    x = x_values[0]
    y = y_values[0]

    for i in range(1, len(x_values)):
        x = np.concatenate((x, x_values[i]), axis=0)
        y = np.concatenate((y, y_values[i]), axis=0)

    del x_values
    del y_values

    return x, y


def time_to_one_hot(dataset):
    new_dataset = dataset.values

    for i in range(new_dataset.shape[0]):
        new_dataset[i][1] = new_dataset[i][1].split(' ')[1][:5]

    new_dataset = pd.DataFrame(new_dataset[:, 1], columns=['tstp'])

    dummies = pd.get_dummies(new_dataset['tstp'], prefix='time')

    return dummies


def get_low_anomalies(data):
    low_anomalies = []

    previous_date = datetime.strptime(str(data.iloc[0]['tstp'])[:-1], '%Y-%m-%d %H:%M:%S.%f')

    for i in range(1, len(data)):
        current_date = datetime.strptime(str(data.iloc[i]['tstp'])[:-1], '%Y-%m-%d %H:%M:%S.%f')

        difference = (current_date - previous_date).total_seconds() / 60

        if difference != 30.:
            difference_slots = difference / 30
            if difference_slots < 1:
                low_anomalies.append({'index': i,
                                      'state': 'less',
                                      'previous_date': previous_date,
                                      'diff': difference_slots})

        previous_date = current_date

    for i in range(len(low_anomalies) - 1):
        if low_anomalies[i]['diff'] + low_anomalies[i + 1]['diff'] == 1:
            energy1 = 0
            energy2 = 0
            if data.iloc[low_anomalies[i]['index']]['energy(kWh/hh)'] != 'Null':
                energy1 = float(data.iloc[low_anomalies[i]['index']]['energy(kWh/hh)'])
            if data.iloc[low_anomalies[i+1]['index']]['energy(kWh/hh)'] != 'Null':
                energy2 = float(data.iloc[low_anomalies[i+1]['index']]['energy(kWh/hh)'])

            low_anomalies[i]['energy'] = energy1 + energy2
            del(low_anomalies[i + 1])
            i += 1

    return low_anomalies


def get_high_anomalies(data):
    high_anomalies = []

    previous_date = datetime.strptime(str(data.iloc[0]['tstp'])[:-1], '%Y-%m-%d %H:%M:%S.%f')

    for i in range(1, len(data)):
        current_date = datetime.strptime(str(data.iloc[i]['tstp'])[:-1], '%Y-%m-%d %H:%M:%S.%f')

        difference = (current_date - previous_date).total_seconds() / 60

        if difference != 30.:
            difference_slots = difference / 30

            if difference_slots > 1:
                high_anomalies.append({'index': i,
                                       'state': 'more',
                                       'previous_date': previous_date,
                                       'diff': difference_slots})
        previous_date = current_date

    return high_anomalies


class ToTensor(object):
    def __call__(self, sample):
        x, y = sample['x'], sample['y']

        new_sample = {'x': torch.tensor(x, dtype=torch.float32), 'y': torch.tensor(y, dtype=torch.float32)}

        return new_sample




