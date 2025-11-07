import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
import os

# ========== HIPERPARÂMETROS OTIMIZADOS ==========
HYPERPARAMS = {
    'seq_length': 15,          # Aumentado de 10 para capturar mais histórico
    'hidden_size': 256,        # Aumentado de 128 para mais capacidade
    'num_layers': 3,           # Aumentado de 2 para mais profundidade
    'learning_rate': 0.0005,   # Reduzido de 0.001 para melhor convergência
    'batch_size': 16,          # Reduzido de 32 para melhor generalização
    'epochs': 200,             # Aumentado de 100 para melhor treinamento
    'dropout': 0.3,            # Adicionar dropout para evitar overfitting
    'weight_decay': 1e-5       # Regularização L2
}

# ========== CLASSES ==========
class MegaSenaDataset(Dataset):
    def __init__(self, X, y_onehot):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_onehot, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=60, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_classes * 6)
        self.num_classes = num_classes
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.view(-1, 6, self.num_classes)

# ========== CARREGAMENTO DE DADOS ==========
print("Carregando dados...")
df = pd.read_csv('results.csv')
df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')
df = df.sort_values('Concurso').reset_index(drop=True)

df['draw'] = df[['N1','N2','N3','N4','N5','N6']].apply(lambda row: sorted(row.tolist()), axis=1)
draws = df['draw'].tolist()

# ========== FEATURE ENGINEERING ==========
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

    most_common = np.argsort(freq)[-3:] + 1
    least_common = np.argsort(freq)[:3] + 1

    features = np.concatenate([
        freq / (i + 1), 
        np.array(days_delayed) / 365, 
        pair_feature, 
        trio_feature, 
        np.eye(60)[most_common-1].sum(axis=0)[:10], 
        np.eye(60)[least_common-1].sum(axis=0)[:10]
    ])
    
    return features

print("Extraindo features...")
features_list = []
dates = df['Data'].tolist()
for i in range(1, len(draws)):
    feat = calculate_features(draws[:i], dates[:i], i)
    features_list.append(feat)

features_array = np.array(features_list)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_array)

# ========== PREPARAÇÃO DOS DADOS ==========
seq_length = HYPERPARAMS['seq_length']
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

# ========== MODELO ==========
print("\nInicializando modelo com hiperparâmetros otimizados:")
for key, value in HYPERPARAMS.items():
    print(f"  {key}: {value}")

input_size = features_scaled.shape[1]
model = LSTMPredictor(
    input_size, 
    hidden_size=HYPERPARAMS['hidden_size'],
    num_layers=HYPERPARAMS['num_layers'],
    dropout=HYPERPARAMS['dropout']
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=HYPERPARAMS['learning_rate'],
    weight_decay=HYPERPARAMS['weight_decay']
)

# Learning rate scheduler para melhor convergência
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

dataset = MegaSenaDataset(X, y)
loader = DataLoader(dataset, batch_size=HYPERPARAMS['batch_size'], shuffle=True)

# ========== TREINAMENTO ==========
print("\nIniciando treinamento...\n")
training_history = []

model.train()
for epoch in range(HYPERPARAMS['epochs']):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    training_history.append({
        'epoch': epoch + 1,
        'loss': avg_loss
    })
    
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{HYPERPARAMS["epochs"]}: Loss = {avg_loss:.6f}')

# ========== AVALIAÇÃO ==========
print("\nAvaliando modelo nos últimos 100 sorteios...\n")
model.eval()

results = []
with torch.no_grad():
    test_X = torch.tensor(X[-100:], dtype=torch.float32)
    preds = model(test_X).sigmoid()
    hits_4plus = 0
    hits_5plus = 0
    hits_6 = 0
    
    for i, pred in enumerate(preds):
        predicted_nums = []
        for pos in range(6):
            num_idx = torch.argmax(pred[pos])
            predicted_nums.append(num_idx.item() + 1)
        
        predicted_nums = sorted(set(predicted_nums))
        if len(predicted_nums) < 6:
            remaining = [n for n in range(1, 61) if n not in predicted_nums]
            predicted_nums.extend(remaining[:6-len(predicted_nums)])
        predicted_nums = sorted(predicted_nums[:6])

        real_nums = sorted([int(n) for n in y_draws[-100:][i]])
        hits = len(set(predicted_nums) & set(real_nums))
        
        if hits >= 4:
            hits_4plus += 1
        if hits >= 5:
            hits_5plus += 1
        if hits == 6:
            hits_6 += 1
        
        results.append({
            'sorteio_index': len(draws) - 100 + i,
            'concurso': df.iloc[len(draws) - 100 + i]['Concurso'],
            'data': df.iloc[len(draws) - 100 + i]['Data'].strftime('%d/%m/%Y'),
            'previsto_1': predicted_nums[0],
            'previsto_2': predicted_nums[1],
            'previsto_3': predicted_nums[2],
            'previsto_4': predicted_nums[3],
            'previsto_5': predicted_nums[4],
            'previsto_6': predicted_nums[5],
            'real_1': real_nums[0],
            'real_2': real_nums[1],
            'real_3': real_nums[2],
            'real_4': real_nums[3],
            'real_5': real_nums[4],
            'real_6': real_nums[5],
            'acertos': hits
        })

# ========== PREVISÃO PRÓXIMO SORTEIO ==========
def predict_next(last_draws_features):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(last_draws_features[-seq_length:]).unsqueeze(0).float()
        pred = model(input_seq).sigmoid()
        predicted = []
        for pos in range(6):
            num = torch.argmax(pred[0, pos]) + 1
            predicted.append(num.item())
        
        predicted = sorted(set(predicted))
        if len(predicted) < 6:
            remaining = [n for n in range(1, 61) if n not in predicted]
            predicted.extend(remaining[:6-len(predicted)])
        return sorted(predicted[:6])

next_prediction = predict_next(features_scaled[-seq_length:])

# ========== EXPORTAÇÃO PARA CSV ==========
print("\nExportando resultados para CSV...")

# 1. Resultados da avaliação
results_df = pd.DataFrame(results)
results_df.to_csv('resultados_avaliacao.csv', index=False, encoding='utf-8-sig')
print(f"✓ Salvos resultados de avaliação em 'resultados_avaliacao.csv'")

# 2. Histórico de treinamento
training_df = pd.DataFrame(training_history)
training_df.to_csv('historico_treinamento.csv', index=False, encoding='utf-8-sig')
print(f"✓ Salvo histórico de treinamento em 'historico_treinamento.csv'")

# 3. Estatísticas gerais
stats = {
    'total_sorteios_avaliados': len(results),
    'acertos_4_ou_mais': hits_4plus,
    'percentual_4_mais': (hits_4plus / len(results)) * 100,
    'acertos_5_ou_mais': hits_5plus,
    'percentual_5_mais': (hits_5plus / len(results)) * 100,
    'acertos_6': hits_6,
    'percentual_6': (hits_6 / len(results)) * 100,
    'loss_final': training_history[-1]['loss'],
    'previsao_proximo': str(next_prediction)
}

stats_df = pd.DataFrame([stats])
stats_df.to_csv('estatisticas_modelo.csv', index=False, encoding='utf-8-sig')
print(f"✓ Salvas estatísticas gerais em 'estatisticas_modelo.csv'")

# 4. Próxima previsão
next_pred_df = pd.DataFrame([{
    'data_previsao': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
    'numero_1': next_prediction[0],
    'numero_2': next_prediction[1],
    'numero_3': next_prediction[2],
    'numero_4': next_prediction[3],
    'numero_5': next_prediction[4],
    'numero_6': next_prediction[5]
}])
next_pred_df.to_csv('proxima_previsao.csv', index=False, encoding='utf-8-sig')
print(f"✓ Salva próxima previsão em 'proxima_previsao.csv'")

# ========== RESUMO FINAL ==========
print("\n" + "="*70)
print("RESUMO DA AVALIAÇÃO")
print("="*70)
print(f"Total de sorteios avaliados: {len(results)}")
print(f"Acertos de 4+: {hits_4plus} ({hits_4plus/len(results)*100:.2f}%)")
print(f"Acertos de 5+: {hits_5plus} ({hits_5plus/len(results)*100:.2f}%)")
print(f"Acertos de 6: {hits_6} ({hits_6/len(results)*100:.2f}%)")
print(f"\nLoss final: {training_history[-1]['loss']:.6f}")
print(f"\nPrevisão para próximo sorteio: {next_prediction}")
print("="*70)
print("\n✓ Todos os resultados foram exportados para arquivos CSV!")