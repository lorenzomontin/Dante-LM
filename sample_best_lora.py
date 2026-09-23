import re
import urllib.request

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "LorenzoDeMattei/GePpeTto"
URL = "https://dmf.unicatt.it/~della/pythoncourse18/commedia.txt"
BLOCK_SIZE = 128
SEED = 1  # matches one of the 3 finalist seeds already reported
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)  # on CPU, 1000 steps will take a while — see note below

# ---------- data (same cleaning + split as GPT.ipynb / the ablation notebook) ----------
title_lines = {"LA DIVINA COMMEDIA", "di Dante Alighieri", "INFERNO", "PURGATORIO", "PARADISO"}
canto_header_re = re.compile(r"^(Inferno|Purgatorio|Paradiso):\s*Canto\s+[IVXLCDM]+\s*$")


def clean_editorial_lines(text):
    return "\n".join(
        line for line in text.splitlines()
        if line.strip() not in title_lines and not canto_header_re.match(line.strip())
    )


with urllib.request.urlopen(URL) as response:
    dante_text = clean_editorial_lines(response.read().decode("utf-8"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data = torch.tensor(tokenizer.encode(dante_text), dtype=torch.long)
n1, n2 = int(0.9 * len(data)), int(0.95 * len(data))
train, val, test = data[:n1], data[n1:n2], data[n2:]


def get_batch(batch_size, block_size=BLOCK_SIZE):
    ix = torch.randint(0, len(train) - block_size, (batch_size,))
    return torch.stack([train[i : i + block_size] for i in ix])


@torch.no_grad()
def full_pass_loss(model, split, block_size=BLOCK_SIZE, chunk_batch=16):
    model.eval()
    data_split = val if split == "val" else test
    n = (len(data_split) // block_size) * block_size
    windows = data_split[:n].view(-1, block_size)
    total = 0.0
    for i in range(0, len(windows), chunk_batch):
        xb = windows[i : i + chunk_batch].to(device)
        total += model(input_ids=xb, labels=xb).loss.item() * xb.shape[0]
    model.train()
    return total / len(windows)


# ---------- train the best config: r=8, all-linear targets ----------
torch.manual_seed(SEED)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj", "c_fc"],
    fan_in_fan_out=True,
)
model = get_peft_model(model, lora_config)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

for step in range(1000):
    model.train()
    xb = get_batch(8).to(device)
    loss = model(input_ids=xb, labels=xb).loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (step + 1) % 100 == 0:
        print(f"step {step + 1:4d} | train {loss.item():.4f} | "
              f"val {full_pass_loss(model, 'val'):.4f}")

model.save_pretrained("geppetto-dante-lora-all-linear")

# ---------- sample ----------
model.eval()
for prompt in ["Amor che ne la mente mi ragiona", "Nel mezzo del cammin di nostra vita"]:
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids, max_new_tokens=150, do_sample=True,
            temperature=0.8, top_k=50, pad_token_id=tokenizer.eos_token_id,
        )
    print("=" * 60)
    print(tokenizer.decode(out[0], skip_special_tokens=True))