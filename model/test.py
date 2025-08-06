import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

seq_len = 128
batch_size = 32
learning_rate = 3e-4
num_epochs = 10
warmup_steps = 500
checkpoint_every = 2
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    tokens = tokenizer(example["text"], truncation=True, padding='max_length', max_length=seq_len, return_tensors='pt')
    return {
        "input_ids": tokens["input_ids"].squeeze(0),
        "labels": tokens["input_ids"].squeeze(0)
    }

raw_dataset = load_dataset("roneneldan/TinyStories", split="train")

if os.path.exists("tokenized_tinystories"):
    tokenized_dataset = load_from_disk("tokenized_tinystories")
else:
    tokenized_dataset = raw_dataset.map(tokenize_function)
    tokenized_dataset.save_to_disk("tokenized_tinystories")

def format_for_torch(example):
    return {
        "input_ids": torch.tensor(example["input_ids"]),
        "labels": torch.tensor(example["input_ids"])
    }

tokenized_dataset.set_transform(format_for_torch)

train_size = int(0.9 * len(tokenized_dataset))
val_size = len(tokenized_dataset) - train_size
train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

from model import DPS, DPS_Config

config = DPS_Config()
config.max_batch_size = batch_size
config.learning_rate = learning_rate
config.warmup_steps = warmup_steps
config.num_epochs = num_epochs
config.max_seq_len = seq_len
model = DPS(config).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0)
)

loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        progress_bar.set_postfix(train_loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

    if (epoch + 1) % checkpoint_every == 0:
        ckpt_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()