# Setup and Imports
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hyperparameters
seq_len = 128
batch_size = 96
learning_rate = 3e-4
num_epochs = 20
warmup_steps = 500
checkpoint_every = 2
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Dataset Loading
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=seq_len,
    )
    return {
        "input_ids": tokens["input_ids"],
        "labels": tokens["input_ids"]
    }

if os.path.exists("tokenized_fineweb"):
    tokenized_dataset = load_from_disk("tokenized_fineweb")
else:
    raw_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    tokenized_dataset = raw_dataset.map(tokenize_function, remove_columns=["text"])
    tokenized_dataset.save_to_disk("tokenized_fineweb")

def format_for_torch(example):
    return {
        "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
        "labels": torch.tensor(example["labels"], dtype=torch.long)
    }

tokenized_dataset.set_transform(format_for_torch)

# Dataset Split
train_size = int(0.9 * len(tokenized_dataset))
val_size = len(tokenized_dataset) - train_size
train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, val_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)

# Model and Config
from model import DPS, DPS_Config

config = DPS_Config()
config.max_batch_size = batch_size
config.learning_rate = learning_rate
config.warmup_steps = warmup_steps
config.num_epochs = num_epochs
config.max_seq_len = seq_len

model = DPS(config).to(device)
# model = torch.compile(model)

# Optimizer and Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0))

# Loss & Mixed Precision Tools
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
scaler = GradScaler()

# Tracking losses
train_losses, val_losses = [], []

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for batch in progress_bar:
        # start_time = time.time()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", torch.float16):
            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_train_loss += loss.item()
        progress_bar.set_postfix(train_loss=loss.item())

        # step_time = time.time() - start_time
        # print(f"⏱️ Step Time: {step_time:.2f}s")

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", torch.float16):
                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"\n✅ Epoch {epoch + 1} Completed: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    if (epoch + 1) % checkpoint_every == 0:
        ckpt_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

# Plotting
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()