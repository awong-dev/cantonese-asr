import json
import re

chars_to_ignore_regex = (r'[\,\?\.\!\-\;\:"\"\%\'\"\�\．\⋯\！\－\：\–\。\》'
                         r'\,\）\,\？\；\～\~\…\︰\，'
                         r'\（\」\‧\《\﹔\、\—\／\,\「\﹖\·\']')

# Load your training data
from datasets import load_dataset
dataset_dir = 'data/cv-corpus-25.0-2026-03-09/yue'
train_dataset = load_dataset('csv', data_files=f'{dataset_dir}/train.tsv', delimiter="\t", split="train")
test_dataset = load_dataset('csv', data_files=f'{dataset_dir}/test.tsv', delimiter="\t", split="train")

# Extract all unique characters
def extract_chars(batch):
    text = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return {"chars": list(set(text))}

train_chars = train_dataset.map(extract_chars)
test_chars = test_dataset.map(extract_chars)

all_chars = set()
for row in train_chars:
    all_chars.update(row["chars"])
for row in test_chars:
    all_chars.update(row["chars"])

# Build vocab dict
vocab = {c: i for i, c in enumerate(sorted(all_chars))}
vocab["[UNK]"] = len(vocab)
vocab["[PAD]"] = len(vocab)

# Save it
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
