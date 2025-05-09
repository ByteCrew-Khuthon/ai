from dataset import AudioMelDataset
from model import CustomModel
import torch
import torchaudio
from tqdm import tqdm
device = torch.device("mps")

from glob import glob
file_list = glob("./dataset/*.wav")

file_new_list = []
for f in file_list :
    try :
        waveform, sr = torchaudio.load(f)
        file_new_list.append(f)
    except Exception :
        continue
    

import random
train_list = random.sample(file_new_list, int(len(file_new_list) * 0.8))
val_list = [file for file in file_new_list if file not in train_list]


from torch.utils.data import DataLoader
import torch.nn as nn

train_dataset = AudioMelDataset(train_list)
val_dataset = AudioMelDataset(val_list)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = CustomModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


model.to(device)
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        x = batch.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() # accumulate

    return total_loss / len(loader.dataset)

# 검증 함수
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            x = batch.to(device)
            output = model(x)
            loss = criterion(output, x)
            total_loss += loss.item()

    return total_loss / len(loader.dataset)

num_epochs = 15
best_val = 100
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss = validate(model, val_loader, criterion, device)
    if best_val > val_loss :
        best_val = val_loss
        torch.save(model.state_dict(), "./best.pth")
        print("Best model saved ...")
    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")