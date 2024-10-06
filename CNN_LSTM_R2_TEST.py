#NAN값이 많나?

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import random

# 1. 데이터 불러오기
file_path = "C:/Users/user/Desktop/sampling_exel data/10.03protocol/MAKE DATA/추정연/all data.xlsx"
data = pd.read_excel(file_path)

# features = ['Heart Rate', 'R_knee_angle_vel', 'L_knee_angle_vel',
#             'R_knee_x_acc', 'L_knee_x_acc', 'R_knee_y_acc', 'L_knee_y_acc',
#             'R_knee_z_acc', 'L_knee_z_acc', 'Amplitude']
# 여러 입력값 사용
features = ['Heart Rate','R_knee_angle_vel', 'L_knee_angle_vel', 'Amplitude']  # 입력값을 여러 개 사용
target = ['VO2']  # 타겟 값

# 입력 데이터 (X)와 타겟 데이터 (y) 추출
X = data[features].values
y = data[target].values

# 3. 시퀀스 나누기 (3초 단위로, 300 타임스텝)
sequence_length = 300  # 100Hz 데이터로 3초에 해당


# 랜덤 시드 설정 함수
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # GPU 사용 시 시드 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# 시퀀스 데이터를 만드는 함수
def create_sequences(data, target, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :]  # 여러 입력값을 처리할 수 있도록 수정
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


X_seq, y_seq = create_sequences(X, y, sequence_length)

# NaN 값 처리 추가
X_seq = np.nan_to_num(X_seq)  # NaN을 0으로 대체
y_seq = np.nan_to_num(y_seq)

# 다시 텐서로 변환
X_seq = torch.tensor(X_seq, dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32)

# 4. 학습 데이터와 검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# DataLoader로 데이터셋 준비
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 5. CUDA 사용 설정 (GPU가 사용 가능한지 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 6. CNN+LSTM 모델 정의 (다중 입력 지원)
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CNN_LSTM_Model, self).__init__()

        # CNN layers (1D Convolution for time-series data)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layers
        self.fc = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # CNN에 맞게 입력 데이터 변환
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # LSTM에 맞게 데이터 차원 변환
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # 마지막 타임스텝의 출력값 사용

        # Fully connected layers
        out = self.relu(lstm_out)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


# 모델 생성
input_size = X_seq.shape[2]
hidden_size = 128
output_size = 1

model = CNN_LSTM_Model(input_size, hidden_size, output_size).to(device)

# 7. 손실 함수 및 옵티마이저 설정 (MAE 사용)
criterion = nn.L1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. 학습률 스케줄러 설정
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 9. 모델 학습 함수 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        scheduler.step()

        print(
            f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')


# R² 계산 함수 추가
def calculate_r2_score(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    r2 = r2_score(all_targets, all_preds)
    return r2


# 10. 모델 학습
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50)

# 11. 학습 완료 후 R² 계산 및 출력
r2 = calculate_r2_score(model, val_loader)
print(f'검증 데이터셋에 대한 R² 상관계수: {r2:.4f}')

# 12. 모델 저장
torch.save(model.state_dict(), "cnn_lstm_ve_prediction_model.pth")
