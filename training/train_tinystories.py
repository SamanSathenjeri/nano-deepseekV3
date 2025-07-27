import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
import os

from model import DPS, DPS_Config  # <-- your model and config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Hyperparameters ===
seq_len = 128
batch_size = 8
epochs = 5
lr = 1e-4
eval_every = 1

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# === Load TinyStories dataset ===
print("🔄 Loading TinyStories...")
raw_dataset = load_dataset("roneneldan/TinyStories", split="train")

# === Tokenization function ===
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=seq_len)

print("🔠 Tokenizing...")
tokenized = raw_dataset.map(tokenize_function, batched=True)
tokenized.set_format(type="torch", columns=["input_ids"])

# === PyTorch Dataset wrapper ===
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data["input_ids"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return x, y

# === Prepare dataset ===
full_dataset = TextDataset(tokenized)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# === Initialize Model ===
print("🚀 Initializing DPS model")
config = DPS_Config() 
config.vocab_size = tokenizer.vocab_size
config.max_seq_len = seq_len
model = DPS(config).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

train_losses = []
val_losses = []

# === Training Loop ===
print("\n🧪 Starting Training on TinyStories")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print('x')

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch {epoch:03} | Train Loss: {avg_train_loss:.4f}", end='')

    # === Validation ===
    if epoch % eval_every == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.contiguous().view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f" | Val Loss: {avg_val_loss:.4f}")
    else:
        print()

# === Save Model Checkpoint ===
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/dps_tinystories_final.pt")
print("\n💾 Final model saved to checkpoints/dps_tinystories_final.pt")

# === Plot and Save Loss Curve ===
plt.plot(train_losses, label="Train Loss")
if val_losses:
    plt.plot(range(eval_every, epochs + 1, eval_every), val_losses, label="Val Loss")
plt.title("TinyStories Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("visualizations/tinystories_loss_curve.png")
print("📈 Loss curve saved to tinystories_loss_curve.png")
