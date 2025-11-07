import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

#classes
class MegaSenaDataset(Dataset):
    def __init__(self, X, y_onehot):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_onehot, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=60):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes * 6)
        self.num_classes = num_classes
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.view(-1, 6, self.num_classes)

#carregamento
df = pd.read_csv('results.csv')
df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
df = df.sort_values('Concurso').reset_index(drop=True)

df['draw'] = df[['N1','N2','N3','N4','N5','N6']].apply(lambda row: sorted(row.tolist()), axis=1)
draws = df['draw'].tolist()

def calculate_features(draws_up_to_i, dates_up_to_i, i):
    freq = np.zeros(60)
    last_appearance = np.full(60, dates_up_to_i[0])
    for j in range(i):
        for num in draws_up_to_i[j]:
            freq[num-1] += 1
            last_appearance[num-1] = dates_up_to_i[j]
    
    days_delayed = [(dates_up_to_i[-1] - last_appearance[k]).days for k in range(60)]

    recent_draws = draws_up_to_i[max(0, i-50):i]
    pair_freq = defaultdict(int)
    for draw in recent_draws:
        for pair in combinations(sorted(draw), 2):
            pair_freq[pair] += 1
    top_pairs = sorted(pair_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    pair_feature = np.array([p[1] for p in top_pairs])

    trio_freq = defaultdict(int)
    for draw in recent_draws:
        for trio in combinations(sorted(draw), 3):
            trio_freq[trio] += 1
    top_trios = sorted(trio_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    trio_feature = np.array([p[1] for p in top_trios])

    most_common =  np.argsort(freq)[-3:] + 1
    least_common = np.argsort(freq)[:3] + 1

    features = np.concatenate([freq /(i+1), 
                               np.array(days_delayed) / 365, 
                               pair_feature, trio_feature, 
                               np.eye(60)[most_common-1].sum(axis=0)[:10], 
                               np.eye(60)[least_common-1].sum(axis=0)[:10]])
    
    return features

features_list = []
dates = df['Data'].tolist()
for i in range(1, len(draws)):
    feat = calculate_features(draws[:i], dates[:i], i)
    features_list.append(feat)

features_array = np.array(features_list)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_array)

seq_length = 10
X, y = [], []
for i in range(seq_length, len(features_scaled)):
    X.append(features_scaled[i-seq_length:i])
    y.append(draws[i])

X = np.array(X)
y_draws = np.array(y)

y = np.zeros((len(y), 6, 60))
for i, draw in enumerate(y_draws):
    for j, num in enumerate(draw):
        y[i, j, num-1] = 1

#modelo
input_size = features_scaled.shape[1]
model = LSTMPredictor(input_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = MegaSenaDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

epochs = 100
model.train()
for epoch in range(epochs):
    total_loss=0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}')

#avaliacao
model.eval()
with torch.no_grad():
    test_X = torch.tensor(X[-100:], dtype=torch.float32)
    preds = model(test_X).sigmoid()
    hits_4plus = 0
    for i, pred in enumerate(preds):
        predicted_nums = []
        for pos in range(6):
            num_idx = torch.argmax(pred[pos])
            predicted_nums.append(num_idx.item() + 1) 
        predicted_nums = sorted(set(predicted_nums))[:6]

        real_nums = [np.argmax(pos) + 1 for pos in y[i]]
        hits = len(set(predicted_nums) & set(real_nums))
        if hits >= 4:
            hits_4plus += 1
        print(f'Sorteio {i}: Previsto {predicted_nums}, Real {real_nums}, Acertos: {hits}')
print(f'% de vezes com 4+ acertos: {hits_4plus / len(preds) * 100:.2f}')


#previsao
def predict_next(last_draws_features):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(last_draws_features[-seq_length:]).unsqueeze(0).float()
        pred = model(input_seq).sigmoid()
        predicted = []
        for pos in range(6):
            num = torch.argmax(pred[0, pos]) + 1
            predicted.append(num.item())
        return sorted(set(predicted))[:6]

next_prediction = predict_next(features_scaled[-seq_length:])
print(f'Previsão para próximo: {next_prediction}') 


