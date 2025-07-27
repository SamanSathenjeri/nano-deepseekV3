import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from model import DPS, DPS_Config

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 128
batch_size = 4
num_epochs = 100
print_interval = 10
plot_path = "small_batch_loss_curve.png"

# --- Load small data subset ---
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:8]")

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=seq_len,
    )

dataset = dataset.map(tokenize_fn, remove_columns=["text"])
input_ids = torch.tensor(dataset["input_ids"])  # [8, seq_len]
labels = input_ids.clone()

# --- Build DataLoader ---
dataset_tensor = torch.utils.data.TensorDataset(input_ids, labels)
loader = torch.utils.data.DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True)

# --- Model Setup ---
config = DPS_Config()
config.vocab_size = tokenizer.vocab_size 
config.max_seq_len = seq_len
model = DPS(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# --- Training Loop ---
train_losses = []

print("\n🧪 Starting Overfit on Small Batch")
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)  # [B, T, V]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    train_losses.append(avg_loss)

    if epoch % print_interval == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f}")

# --- Plot Loss Curve ---
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overfitting Small Batch")
plt.grid(True)
plt.legend()
plt.savefig(plot_path)
plt.close()
print(f"\n📈 Loss curve saved to {plot_path}")

# --- Evaluation on Batch ---
model.eval()
with torch.no_grad():
    x_batch, y_batch = next(iter(loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    preds = model(x_batch)
    predicted_ids = torch.argmax(preds, dim=-1)

    for i in range(batch_size):
        input_txt = tokenizer.decode(x_batch[i], skip_special_tokens=True)
        target_txt = tokenizer.decode(y_batch[i], skip_special_tokens=True)
        pred_txt = tokenizer.decode(predicted_ids[i], skip_special_tokens=True)

        print(f"\n--- Sample {i+1} ---")
        print("🔎 Input:\n", input_txt[:200], "...")
        print("🎯 Target:\n", target_txt[:200], "...")
        print("🤖 Prediction:\n", pred_txt[:200], "...")

print("\n✅ Overfitting small batch finished!")