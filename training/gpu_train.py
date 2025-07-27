import os, time, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random

from model import DPS, DPS_Config

# ✅ Config
MODEL_NAME = "gpt2"
MAX_LENGTH = 1024
TRAIN_EXAMPLES = 2500   # Increase this to 100k+ later
VAL_EXAMPLES = 5000
BATCH_SIZE = 2  # Adjust based on memory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
EPOCHS = 100
EVAL_EVERY = 5
LOG_PATH = "training_log.json"
MODEL_SAVE_DIR = "dps_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ✅ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ Custom IterableDataset with shuffle and chunking
class TokenChunkDataset(IterableDataset):
    def __init__(self, dataset_stream, tokenizer, max_length):
        self.stream = dataset_stream
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        buffer = []
        for example in self.stream:
            ids = self.tokenizer(example["text"], return_attention_mask=False, truncation=False)["input_ids"]
            buffer.extend(ids)
            while len(buffer) >= self.max_length:
                chunk = buffer[:self.max_length]
                buffer = buffer[self.max_length:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }

# ✅ Load shuffled stream and split
print("🔄 Loading and shuffling stream...")
raw_stream = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
shuffled_stream = raw_stream.shuffle(buffer_size=10000, seed=42)

# ✅ Build train/val splits
def take_samples(stream, count):
    return list(iter(torch.utils.data.DataLoader(
        TokenChunkDataset(stream, tokenizer, MAX_LENGTH),
        batch_size=1
    )))[:count]

# Split the shuffled stream manually
print("🔄 Creating train/val buffers...")
split_stream = iter(shuffled_stream)
train_stream = (next(split_stream) for _ in range(TRAIN_EXAMPLES * 2))
val_stream   = (next(split_stream) for _ in range(VAL_EXAMPLES * 2))

train_samples = take_samples(train_stream, TRAIN_EXAMPLES)
val_samples = take_samples(val_stream, VAL_EXAMPLES)

# ✅ Flatten single-sample batches
train_dataset = [{
    "input_ids": s["input_ids"].squeeze(0),
    "labels": s["labels"].squeeze(0),
} for s in train_samples]

val_dataset = [{
    "input_ids": s["input_ids"].squeeze(0),
    "labels": s["labels"].squeeze(0),
} for s in val_samples]

# ✅ Create Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ✅ Model setup
print("🚀 Initializing DPS model")
config = DPS_Config()
config.vocab_size = tokenizer.vocab_size
config.hidden_size = 128
config.num_heads = 4
config.num_layers = 4
config.intermediate_size = 512
config.device = DEVICE

import gc
torch.cuda.empty_cache()
gc.collect()

model = DPS(config).to(DEVICE)
# summary(model, input_size=(1, MAX_LENGTH))
model = torch.compile(model)
# model.gradient_checkpointing_enable()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ✅ Resume if checkpoint exists
start_epoch = 1
training_log = []
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r") as f:
        training_log = json.load(f)
    last_entry = training_log[-1]
    start_epoch = last_entry["epoch"] + 1
    ckpt_path = os.path.join(MODEL_SAVE_DIR, f"dps_epoch{last_entry['epoch']:03}.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))
        print(f"🔁 Resumed from epoch {last_entry['epoch']}")

# ✅ Training loop
print("🧪 Starting Training")
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        print(f"✅ Epoch {epoch:03} | Val Loss: {avg_val_loss:.4f}")

    # ✅ Save checkpoint
    if epoch % 5 == 0:
      ckpt_path = os.path.join(MODEL_SAVE_DIR, f"dps_epoch{epoch:03}.pt")
      torch.save(model.state_dict(), ckpt_path)
      print(f"💾 Saved: {ckpt_path}")

    # ✅ Log
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
print("🎉 Training Complete")