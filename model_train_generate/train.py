import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_train_generate.data_prep import train_sequences, val_sequences
from model_train_generate.model import ArmAutoencoder

## setup
EPOCHS = 100 # double the amount of passes for our small data set
BATCH_SIZE = 16 # sequences to learn from
LR = 0.001
PATIENCE = 10 # stops if model is not improving

# check if gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tensor conversion
train_tensor = torch.tensor(train_sequences, dtype = torch.float32)
val_tensor = torch.tensor(val_sequences, dtype = torch.float32)

train_loader = DataLoader(TensorDataset(train_tensor), batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(TensorDataset(val_tensor), batch_size = BATCH_SIZE)

# model
model = ArmAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
criterion = nn.MSELoss() # measures reconstruction difference from original data

# training loop
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    for (batch,) in train_loader:
        batch = batch.to(device)

        output, _= model(batch)         # movement reconstruction
        loss = criterion(output, batch) # measures reconstruction accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() # fits weights

    # validation
    model.eval() # evaluation
    val_loss = 0
    with torch.no_grad():
        for (batch,) in val_loader:
            batch = batch.to(device)
            output, _ = model(batch)
            val_loss += criterion(output, batch).item()

    # average losses
    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | train Loss: {train_loss:.4f}")

    # save best model, stop if no improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f" model saved (loss improved)")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"early stopping at epoch {epoch+1} | no improvement for {PATIENCE} epochs")
            break  # exits the epoch loop
