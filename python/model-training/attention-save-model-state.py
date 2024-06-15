import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict, Counter

import json

def save_tokenization(char_to_idx, idx_to_char, filename):
    with open(filename, 'w') as f:
        json.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)

def load_tokenization(filename):
    with open(filename, 'r') as f:
        mappings = json.load(f)
    char_to_idx = {k: int(v) for k, v in mappings['char_to_idx'].items()}
    idx_to_char = {int(k): v for k, v in mappings['idx_to_char'].items()}
    return char_to_idx, idx_to_char

# Load dog names from CSV file
def load_dog_names_from_csv(filename, limit=4500):
    names = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) > 1 and len(names) < limit:
                names.append(row[1].strip())  # Assumes the second column contains the names
    return names

# Create a dataset for the names
class NameDataset(Dataset):
    def __init__(self, names, max_length=12, tokenization_file=None):
        self.names = names
        self.max_length = max_length
        if tokenization_file and os.path.exists(tokenization_file):
            self.char_to_idx, self.idx_to_char = load_tokenization(tokenization_file)
        else:
            # Build the character set dynamically
            char_set = set('*#')
            for name in names:
                char_set.update(name)
            self.char_to_idx = {ch: idx for idx, ch in enumerate(sorted(char_set))}
            self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
            if tokenization_file:
                save_tokenization(self.char_to_idx, self.idx_to_char, tokenization_file)
        self.pad_idx = self.char_to_idx.get(" ", 0)  # Padding index, using 0 for unknown

        print("Character to Index Mapping:", self.char_to_idx)


    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        name = "*" + name + "#"
        print(f"Processing name: {name}")
        name_idx = [self.char_to_idx.get(ch, self.pad_idx) for ch in name]  # Use pad_idx for unknown characters
        print(f"Name indices: {name_idx}")
        name_idx += [self.pad_idx] * (self.max_length + 2 - len(name_idx))  # Pad to max_length
        return torch.tensor(name_idx[:-1]), torch.tensor(name_idx[1:])  # Input and target

# Define the Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128, max_length=12, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = nn.Embedding(max_length + 2, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_length = max_length
        self.vocab_size = vocab_size

    def forward(self, src, tgt):
        if torch.max(src) >= self.vocab_size or torch.max(tgt) >= self.vocab_size:
            print(f"Index out of range: src max {torch.max(src)}, tgt max {torch.max(tgt)}, vocab_size {self.vocab_size}")
            raise ValueError(f"Index out of range: src max {torch.max(src)}, tgt max {torch.max(tgt)}, vocab_size {self.vocab_size}")
        src = self.embedding(src) + self.pos_encoder(torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0), 1))
        tgt = self.embedding(tgt) + self.pos_encoder(torch.arange(0, tgt.size(1)).unsqueeze(0).repeat(tgt.size(0), 1))
        output = self.transformer(src.permute(1, 0, 2), tgt.permute(1, 0, 2))
        return self.fc_out(output.permute(1, 0, 2))

    def generate(self, src, start_token_idx, end_token_idx, idx_to_char, temperature=1.0):
        if torch.max(src) >= self.vocab_size:
            print(f"Index out of range in generate: src max {torch.max(src)}, vocab_size {self.vocab_size}")
            raise ValueError(f"Index out of range in generate: src max {torch.max(src)}, vocab_size {self.vocab_size}")
        src = self.embedding(src) + self.pos_encoder(torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0), 1))
        memory = self.transformer.encoder(src.permute(1, 0, 2))

        generated = torch.tensor([start_token_idx]).unsqueeze(0)
        for _ in range(self.max_length):
            tgt = self.embedding(generated) + self.pos_encoder(torch.arange(0, generated.size(1)).unsqueeze(0).repeat(generated.size(0), 1))
            output = self.transformer.decoder(tgt.permute(1, 0, 2), memory)
            logits = self.fc_out(output[-1, :, :]) / temperature
            next_token = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), num_samples=1).squeeze(1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            if next_token.item() == end_token_idx:
                break
        return generated

# Custom collate function to pad sequences
def collate_fn(batch):
    max_length = max(len(item[0]) for item in batch)
    inputs = []
    targets = []
    for input_seq, target_seq in batch:
        padded_input = torch.cat([input_seq, torch.zeros(max_length - len(input_seq), dtype=torch.long)])
        padded_target = torch.cat([target_seq, torch.zeros(max_length - len(target_seq), dtype=torch.long)])
        inputs.append(padded_input)
        targets.append(padded_target)
    return torch.stack(inputs), torch.stack(targets)

# Training the model
def train_model(model, dataloader, epochs=10, model_save_path='transformer_model_entire.pth'):
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for src, tgt in dataloader:
            if torch.max(src) >= model.vocab_size or torch.max(tgt) >= model.vocab_size:
                print(f"Index out of range during training: src max {torch.max(src)}, tgt max {torch.max(tgt)}, vocab_size {model.vocab_size}")
                continue  # Skip this batch
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.view(-1, model.vocab_size), tgt[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    # Save the entire model
    torch.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")

# Load the entire model
def load_model(model_save_path='transformer_model_entire.pth'):
    if os.path.exists(model_save_path):
        model = torch.load(model_save_path)
        print(f"Model loaded from {model_save_path}")
        return model
    else:
        print(f"No saved model found at {model_save_path}")
        return None

if __name__ == "__main__":
    # Load names and create dataset
    csv_filename = 'kaggle-dog-name-frequencies.csv'
    tokenization_file = 'tokenization.json'
    names = load_dog_names_from_csv(csv_filename, limit=100)  # Use a smaller subset of data
    dataset = NameDataset(names, tokenization_file=tokenization_file)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)  # Smaller batch size

    # Create the model
    model = TransformerModel(vocab_size=len(dataset.char_to_idx), pad_idx=dataset.pad_idx)
    
    # Load the model state if it exists
    loaded_model = load_model()
    if loaded_model:
        model = loaded_model

    # Train the model
    train_model(model, dataloader)

    # Generate a new name
    model.eval()
    with torch.no_grad():
        src = torch.tensor([[dataset.char_to_idx["*"]]])
        generated_name_idx = model.generate(src, dataset.char_to_idx["*"], dataset.char_to_idx["#"], idx_to_char=dataset.idx_to_char, temperature=1.0)
        generated_name = "".join([dataset.idx_to_char[idx.item()] for idx in generated_name_idx[0][1:-1]])
        print("Generated Dog Name:", generated_name)

