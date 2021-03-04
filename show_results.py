import numpy as np
import pandas as pd
import plotly.express as px


epochs = [i + 1 for i in range(30)]

train_loss_v1 = np.load('results/avg_train_losses_sum_5nodes_k2.npy')
val_loss_v1 = np.load('results/avg_val_losses_sum_5nodes_k2.npy')
test_loss_v1 = np.load('results/avg_test_losses_sum_5nodes_k2.npy')

train_loss_v2 = np.load('results/avg_train_losses_target_15nodes_k3.npy')
val_loss_v2 = np.load('results/avg_val_losses_target_15nodes_k3.npy')
test_loss_v2 = np.load('results/avg_test_losses_target_15nodes_k3.npy')

df1 = pd.DataFrame(columns=['epoch', 'value', 'type'])

for i in range(30):
    df1 = df1.append({'epoch': i + 1, 'value': train_loss_v1[i], 'type': 'TrainLoss'}, ignore_index=True)
    df1 = df1.append({'epoch': i + 1, 'value': val_loss_v1[i], 'type': 'ValLoss'}, ignore_index=True)
    df1 = df1.append({'epoch': i + 1, 'value': test_loss_v1[i], 'type': 'TestLoss'}, ignore_index=True)

fig = px.line(df1, x='epoch', y='value', color='type')
fig.show()

df2 = pd.DataFrame(columns=['epoch', 'value', 'type'])

for i in range(30):
    df2 = df2.append({'epoch': i + 1, 'value': train_loss_v2[i], 'type': 'TrainLoss'}, ignore_index=True)
    df2 = df2.append({'epoch': i + 1, 'value': val_loss_v2[i], 'type': 'ValLoss'}, ignore_index=True)
    df2 = df2.append({'epoch': i + 1, 'value': test_loss_v2[i], 'type': 'TestLoss'}, ignore_index=True)

fig = px.line(df2, x='epoch', y='value', color='type')
fig.show()
