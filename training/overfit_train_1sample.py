import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from model import DPS, DPS_Config

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 128
batch_size = 1
num_epochs = 100
print_interval = 10

# --- Load dataset and tokenizer ---
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Select one sample to overfit on
example = dataset[0]["text"]
tokens = tokenizer(example, return_tensors="pt", truncation=True, padding="max_length", max_length=seq_len)
x = tokens.input_ids[:, :-1].to(device)
y = tokens.input_ids[:, 1:].to(device)

# --- Model setup ---
config = DPS_Config()
config.vocab_size = tokenizer.vocab_size 
config.max_seq_len = seq_len
model = DPS(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# --- Training Loop ---
losses = []
print("\n🧪 Starting Overfit on 1 Sample")

for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    logits = model(x)  # (B, T, V)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % print_interval == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

# --- Plot the loss curve ---
plt.plot(losses)
plt.title("Overfit on Single Example")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("overfit_loss_curve.png")
# plt.show()

# --- Evaluate output ---
model.eval()
with torch.no_grad():
    preds = model(x)
    predicted_ids = torch.argmax(preds, dim=-1)
    decoded = tokenizer.decode(predicted_ids[0])

    print("\n🔎 Input:")
    print(tokenizer.decode(x[0]))
    print("\n🎯 Target:")
    print(tokenizer.decode(y[0]))
    print("\n🤖 Prediction:")
    print(decoded)