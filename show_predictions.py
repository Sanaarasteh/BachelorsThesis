import numpy as np
import pandas as pd

import plotly.express as px

epochs = [1, 2]

test_preds_v1 = np.load('results/test_preds_target_15nodes_k3.npy')
true_preds_v1 = np.load('results/true_preds_target_15nodes_k3.npy')

comparison = np.sqrt(np.sum((test_preds_v1 - true_preds_v1) ** 2, axis=1))
candidates = np.argsort(comparison)[:3]

df11 = pd.DataFrame(columns=['epoch', 'value', 'type'])
df12 = pd.DataFrame(columns=['epoch', 'value', 'type'])
df13 = pd.DataFrame(columns=['epoch', 'value', 'type'])

df11 = df11.append({'epoch': 1, 'value': test_preds_v1[candidates[0]][0], 'type': 'Prediction'}, ignore_index=True)
df11 = df11.append({'epoch': 2, 'value': test_preds_v1[candidates[0]][1], 'type': 'Prediction'}, ignore_index=True)
df11 = df11.append({'epoch': 1, 'value': true_preds_v1[candidates[0]][0], 'type': 'GroundTruth'}, ignore_index=True)
df11 = df11.append({'epoch': 2, 'value': true_preds_v1[candidates[0]][1], 'type': 'GroundTruth'}, ignore_index=True)

df12 = df12.append({'epoch': 1, 'value': test_preds_v1[candidates[1]][0], 'type': 'Prediction'}, ignore_index=True)
df12 = df12.append({'epoch': 2, 'value': test_preds_v1[candidates[1]][1], 'type': 'Prediction'}, ignore_index=True)
df12 = df12.append({'epoch': 1, 'value': true_preds_v1[candidates[1]][0], 'type': 'GroundTruth'}, ignore_index=True)
df12 = df12.append({'epoch': 2, 'value': true_preds_v1[candidates[1]][1], 'type': 'GroundTruth'}, ignore_index=True)

df13 = df13.append({'epoch': 1, 'value': test_preds_v1[candidates[2]][0], 'type': 'Prediction'}, ignore_index=True)
df13 = df13.append({'epoch': 2, 'value': test_preds_v1[candidates[2]][1], 'type': 'Prediction'}, ignore_index=True)
df13 = df13.append({'epoch': 1, 'value': true_preds_v1[candidates[2]][0], 'type': 'GroundTruth'}, ignore_index=True)
df13 = df13.append({'epoch': 2, 'value': true_preds_v1[candidates[2]][1], 'type': 'GroundTruth'}, ignore_index=True)


fig1 = px.line(df11, x='epoch', y='value', color='type')
fig1.show()

fig2 = px.line(df12, x='epoch', y='value', color='type')
fig2.show()

fig3 = px.line(df13, x='epoch', y='value', color='type')
fig3.show()



