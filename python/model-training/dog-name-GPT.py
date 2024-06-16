import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import csv
import json
import os
import string

# Load dog names from CSV file
def load_dog_names_from_csv(filename, limit=None):
    names = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) > 1:
                name = row[1].strip().lower()  # Convert to lowercase for consistency
                if len(name) <= 8 and ' ' not in name:  # Filter out long or multi-word names
                    names.append(name)
            if limit and len(names) >= limit:
                break
    return names

# Save and load tokenization mappings
def save_tokenization(char_to_idx, idx_to_char, filename):
    with open(filename, 'w') as f:
        json.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)

def load_tokenization(filename):
    with open(filename, 'r') as f:
        mappings = json.load(f)
    char_to_idx = {k: int(v) for k, v in mappings['char_to_idx'].items()}
    idx_to_char = {int(k): v for k, v in mappings['idx_to_char'].items()}
    return char_to_idx, idx_to_char

# Create a dataset for the names
class NameDataset(Dataset):
    def __init__(self, names, max_length=12, tokenization_file=None):
        self.names = names
        self.max_length = max_length
        if tokenization_file and os.path.exists(tokenization_file):
            self.char_to_idx, self.idx_to_char = load_tokenization(tokenization_file)
        else:
            char_set = set(string.ascii_lowercase + '*# ')  # Only include lowercase alphabetic characters, start token, end token, and space
            self.char_to_idx = {ch: idx for idx, ch in enumerate(sorted(char_set))}
            self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
            self.pad_idx = self.char_to_idx[" "]
            self.start_token_idx = self.char_to_idx.get("*")
            self.end_token_idx = self.char_to_idx.get("#")
            self.vocab_size = len(self.char_to_idx)  # Update vocab_size to include the padding index
            if tokenization_file:
                save_tokenization(self.char_to_idx, self.idx_to_char, tokenization_file)

        self.pad_idx = self.char_to_idx[" "]  # Ensure pad_idx is set correctly
        self.start_token_idx = self.char_to_idx.get("*")
        self.end_token_idx = self.char_to_idx.get("#")
        self.vocab_size = len(self.char_to_idx)  # Update vocab_size to include the padding index

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        name = "*" + name + "#"
        name_idx = [self.char_to_idx.get(ch, self.pad_idx) for ch in name]  # Use pad_idx for unknown characters
        name_idx += [self.pad_idx] * (self.max_length + 2 - len(name_idx))

        # Ensure all indices are within the vocabulary size
        name_idx = [min(idx, self.vocab_size - 1) for idx in name_idx]

        return torch.tensor(name_idx[:-1]), torch.tensor(name_idx[1:])

# Custom collate function to pad sequences
def collate_fn(batch):
    max_length = max(len(item[0]) for item in batch)
    inputs = []
    targets = []
    for input_seq, target_seq in batch:
        padded_input = torch.cat([input_seq, torch.full((max_length - len(input_seq),), dataset.pad_idx, dtype=torch.long)])
        padded_target = torch.cat([target_seq, torch.full((max_length - len(target_seq),), dataset.pad_idx, dtype=torch.long)])
        inputs.append(padded_input)
        targets.append(padded_target)
    return torch.stack(inputs), torch.stack(targets)

# Define the GPT model
class GPTModel(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, max_length=12, dropout=0.1):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = nn.Embedding(max_length, d_model)
        transformer_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(transformer_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_length = max_length
        self.d_model = d_model
        self.pad_idx = pad_idx  # Ensure pad_idx is set correctly
        self.vocab_size = vocab_size  # Add vocab_size attribute

    def forward(self, src, tgt):
        src = self.embedding(src) + self.pos_encoder(torch.arange(src.size(1)).to(src.device))
        tgt = self.embedding(tgt) + self.pos_encoder(torch.arange(tgt.size(1)).to(tgt.device))
        memory = self.transformer_decoder(tgt, src)
        output = self.fc_out(memory)
        return output

    def generate(self, start_token_idx, end_token_idx, idx_to_char, max_length=12, temperature=1.0):
        generated = torch.tensor([[start_token_idx]]).to(next(self.parameters()).device)
        for _ in range(max_length):
            tgt = self.embedding(generated) + self.pos_encoder(torch.arange(generated.size(1)).to(generated.device))
            memory = self.transformer_decoder(tgt, tgt)
            logits = self.fc_out(memory[:, -1, :]) / temperature
            next_token_probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1)
            if next_token.item() >= self.vocab_size:  # Ensure valid token
                next_token = torch.tensor([self.pad_idx]).to(next(self.parameters()).device)
            generated = torch.cat((generated, next_token), dim=1)
            if next_token.item() == end_token_idx:
                break
        return generated

# Train the model
def train_model(model, dataloader, epochs=20, learning_rate=0.001, model_save_path='gpt_model.pth', params_save_path='model_params.json'):
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for src, tgt in dataloader:
            if torch.max(src) >= model.vocab_size or torch.max(tgt) >= model.vocab_size:
                continue
            optimizer.zero_grad()
            try:
                output = model(src, tgt[:, :-1])
                loss = criterion(output.view(-1, model.vocab_size), tgt[:, 1:].contiguous().view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            except IndexError as e:
                continue

    torch.save(model.state_dict(), model_save_path)
    model_params = {
        'vocab_size': model.vocab_size,
        'pad_idx': model.pad_idx,
        'd_model': model.d_model,
        'nhead': model.transformer_decoder.layers[0].self_attn.num_heads,
        'num_layers': len(model.transformer_decoder.layers),
        'dim_feedforward': model.transformer_decoder.layers[0].linear1.out_features,
        'max_length': model.max_length,
        'dropout': model.transformer_decoder.layers[0].dropout.p,
    }
    with open(params_save_path, 'w') as f:
        json.dump(model_params, f)

# Generate dog names
def generate_name(model, start_token_idx, end_token_idx, idx_to_char, temperature=1.0):
    model.eval()
    with torch.no_grad():
        generated_name_idx = model.generate(start_token_idx, end_token_idx, idx_to_char, temperature=temperature)
        generated_name = "".join([idx_to_char[idx.item()] for idx in generated_name_idx[0][1:-1] if idx.item() != model.pad_idx and idx.item() < model.vocab_size])
        return generated_name

if __name__ == "__main__":
    # Load names and create dataset
    csv_filename = 'kaggle-dog-name-frequencies.csv'
    names = load_dog_names_from_csv(csv_filename)
    tokenization_file = 'tokenization.json'

    # Check if tokenization file exists, if not, create it
    if not os.path.exists(tokenization_file):
        dataset = NameDataset(names, tokenization_file=tokenization_file)
    else:
        char_to_idx, idx_to_char = load_tokenization(tokenization_file)
        dataset = NameDataset(names)
        dataset.char_to_idx = char_to_idx
        dataset.idx_to_char = idx_to_char
        dataset.pad_idx = char_to_idx[" "]
        dataset.start_token_idx = char_to_idx.get("*")
        dataset.end_token_idx = char_to_idx.get("#")
        dataset.vocab_size = len(char_to_idx)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Load or create the model
    vocab_size = dataset.vocab_size  # Use the correct vocab size from the dataset
    model = GPTModel(vocab_size=vocab_size, pad_idx=dataset.pad_idx, max_length=dataset.max_length)

    # Load the model if it exists and matches the vocabulary size
    if os.path.exists('gpt_model.pth'):
        state_dict = torch.load('gpt_model.pth')
        if state_dict['embedding.weight'].shape[0] == vocab_size:
            model.load_state_dict(state_dict)
        else:
            print("Saved model vocabulary size does not match the current vocabulary size. Training a new model.")

    # Train the model
    train_model(model, dataloader)

    # Generate a new name
    generated_name = generate_name(model, dataset.start_token_idx, dataset.end_token_idx, idx_to_char=dataset.idx_to_char, temperature=0.8)
    print("Generated Dog Name:", generated_name)
