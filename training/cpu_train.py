import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from model import DPS, DPS_Config

# ✅ Config
SEQ_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-4
EVAL_EVERY = 1
MODEL_SAVE_DIR = "checkpoints"
CACHE_DIR = "./.cache/hf_datasets"
LOG_PATH = os.path.join(MODEL_SAVE_DIR, "training_log.json")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.model_max_length = SEQ_LEN
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"

# ✅ Load dataset with cache
print("🔄 Loading FineWeb-Edu...")
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    split="train[:1%]",
    cache_dir=CACHE_DIR,
    download_mode="reuse_dataset_if_exists"
)
print(f"📚 Loaded {len(dataset)} samples")

# ✅ Tokenize and chunk
def tokenize_and_chunk(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=SEQ_LEN,
        return_tensors="pt"
    )
    return {
        "input_ids": tokens["input_ids"].squeeze(0),
        "attention_mask": tokens["attention_mask"].squeeze(0),
        "labels": tokens["input_ids"].squeeze(0),
    }

tokenized_dataset = dataset.map(tokenize_and_chunk, remove_columns=dataset.column_names)
tokenized_dataset.set_format(type="torch")

# ✅ Train/Val split
train_size = int(0.9 * len(tokenized_dataset))
val_size = len(tokenized_dataset) - train_size
train_dataset, val_dataset = random_split(tokenized_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ✅ Initialize model
print("🚀 Initializing DPS model")
config = DPS_Config()
model = DPS(config).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ✅ Resume training if needed
start_epoch = 1
training_log = []

if os.path.exists(LOG_PATH):
    print("🔁 Resuming from previous training log...")
    with open(LOG_PATH, "r") as f:
        training_log = json.load(f)
    last_entry = training_log[-1]
    start_epoch = last_entry["epoch"] + 1
    latest_ckpt = os.path.join(MODEL_SAVE_DIR, f"dps_epoch{last_entry['epoch']:03}.pt")
    if os.path.exists(latest_ckpt):
        model.load_state_dict(torch.load(latest_ckpt))
        print(f"✅ Loaded checkpoint from epoch {last_entry['epoch']}")

# ✅ Training loop
print("🧪 Starting Training on FineWeb-Edu")
for epoch in range(start_epoch, EPOCHS + 1):
    start_time = time.time()
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)
    print(f"📉 Epoch {epoch:03} | Train Loss: {avg_train_loss:.4f}")

    # ✅ Validation
    avg_val_loss = None
    if epoch % EVAL_EVERY == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"✅ Epoch {epoch:03} | Validation Loss: {avg_val_loss:.4f}")

    # ✅ Save checkpoint
    ckpt_path = os.path.join(MODEL_SAVE_DIR, f"dps_epoch{epoch:03}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"💾 Saved checkpoint: {ckpt_path}")

    # ✅ Log training info
    training_log.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "epoch_time_sec": round(time.time() - start_time, 2),
    })
    with open(LOG_PATH, "w") as f:
        json.dump(training_log, f, indent=2)

# ✅ Final save
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "dps_final.pt"))
print("🎉 Finished Training & Saved Final Model")
