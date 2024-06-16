import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
import json
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

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
            char_set = set('*#')
            for name in names:
                char_set.update(name)
            self.char_to_idx = {ch: idx for idx, ch in enumerate(sorted(char_set))}
            self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
            if tokenization_file:
                save_tokenization(self.char_to_idx, self.idx_to_char, tokenization_file)
        self.pad_idx = self.char_to_idx.get(" ", 0)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        name = "*" + name + "#"
        name_idx = [self.char_to_idx.get(ch, self.pad_idx) for ch in name]
        name_idx += [self.pad_idx] * (self.max_length + 2 - len(name_idx))
        return torch.tensor(name_idx[:-1]), torch.tensor(name_idx[1:])

# Define the custom Transformer Model with batch_first=True
class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model=128, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=512, max_length=12, dropout=0.2):
        super(CustomTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = nn.Embedding(max_length + 2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx  # Ensure pad_idx is set correctly
        self.dropout_value = dropout  # Store dropout value

    def forward(self, src, tgt):
        if torch.max(src) >= self.vocab_size or torch.max(tgt) >= self.vocab_size:
            print(f"Index out of range")
            raise ValueError(f"Index out of range")
        src = self.embedding(src) + self.pos_encoder(torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0), 1).to(src.device))
        tgt = self.embedding(tgt) + self.pos_encoder(torch.arange(0, tgt.size(1)).unsqueeze(0).repeat(tgt.size(0), 1).to(tgt.device))
        memory = self.transformer_encoder(src)
        output = self.transformer_decoder(tgt, memory)
        return self.fc_out(output)

    def generate(self, src, start_token_idx, end_token_idx, idx_to_char, temperature=1.0, max_repeats=2):
        if torch.max(src) >= self.vocab_size:
            print(f"Index out of range in generate")
            raise ValueError(f"Index out of range in generate")
        src = self.embedding(src) + self.pos_encoder(torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0), 1).to(src.device))
        memory = self.transformer_encoder(src)

        generated = torch.tensor([start_token_idx]).unsqueeze(0).to(src.device)
        repeats = defaultdict(int)
        for _ in range(self.max_length):
            tgt = self.embedding(generated) + self.pos_encoder(torch.arange(0, generated.size(1)).unsqueeze(0).repeat(generated.size(0), 1).to(generated.device))
            output = self.transformer_decoder(tgt, memory)
            logits = self.fc_out(output[:, -1, :]) / temperature
            next_token_probs = torch.nn.functional.softmax(logits, dim=-1)

            # Apply repeat penalty
            for token in repeats:
                if repeats[token] >= max_repeats:
                    next_token_probs[0, token] = 0
            next_token = torch.multinomial(next_token_probs, num_samples=1).squeeze(1)

            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            token_id = next_token.item()
            if token_id == end_token_idx:
                break
            repeats[token_id] += 1
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
def train_model(model, dataloader, epochs=20, model_save_path='transformer_model.pth', params_save_path='model_params.json'):
    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate

    for epoch in range(epochs):
        model.train()
        for src, tgt in dataloader:
            if torch.max(src) >= model.vocab_size or torch.max(tgt) >= model.vocab_size:
                print(f"Index out of range during training")
                continue  # Skip this batch
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.view(-1, model.vocab_size), tgt[:, 1:].contiguous().view(-1))
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Loss is NaN or Inf, skipping this batch")
                print(f"src: {src}, tgt: {tgt}")
                continue  # Skip this batch if loss is NaN or Inf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Save the model state and parameters
    torch.save(model.state_dict(), model_save_path)
    model_params = {
        'vocab_size': model.vocab_size,
        'pad_idx': model.pad_idx,
        'd_model': model.embedding.embedding_dim,
        'nhead': model.transformer_encoder.layers[0].self_attn.num_heads,
        'num_encoder_layers': len(model.transformer_encoder.layers),
        'num_decoder_layers': len(model.transformer_decoder.layers),
        'dim_feedforward': model.transformer_encoder.layers[0].linear1.out_features,
        'max_length': model.max_length,
        'dropout': model.dropout_value,
    }
    with open(params_save_path, 'w') as f:
        json.dump(model_params, f)
    print(f"Model and parameters saved to {model_save_path} and {params_save_path}")

# Load the model state
def load_model(tokenization_file, model_save_path='transformer_model.pth', params_save_path='model_params.json'):
    if os.path.exists(model_save_path) and os.path.exists(params_save_path):
        with open(params_save_path, 'r') as f:
            model_params = json.load(f)
        model = CustomTransformerModel(
            vocab_size=model_params['vocab_size'],
            pad_idx=model_params['pad_idx'],
            d_model=model_params['d_model'],
            nhead=model_params['nhead'],
            num_encoder_layers=model_params['num_encoder_layers'],
            num_decoder_layers=model_params['num_decoder_layers'],
            dim_feedforward=model_params['dim_feedforward'],
            max_length=model_params['max_length'],
            dropout=model_params['dropout']  # Use the dropout value directly
        )
        model.load_state_dict(torch.load(model_save_path), strict=False)  # Using strict=False to ignore non-matching keys
        print(f"Model loaded from {model_save_path}")
        return model
    else:
        print(f"No saved model or parameters found at {model_save_path} and {params_save_path}")
        return None

if __name__ == "__main__":
    # Load names and create dataset
    csv_filename = 'kaggle-dog-name-frequencies.csv'
    names = load_dog_names_from_csv(csv_filename)  # Use more data for better results
    tokenization_file = 'tokenization.json'
    dataset = NameDataset(names, tokenization_file=tokenization_file)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)  # Larger batch size

    # Load or create the model
    model = load_model(tokenization_file)
    if model is None:
        model = CustomTransformerModel(vocab_size=len(dataset.char_to_idx), pad_idx=dataset.pad_idx)
    
    # Train the model
    train_model(model, dataloader)

    # Generate a new name
    model.eval()
    with torch.no_grad():
        src = torch.tensor([[dataset.char_to_idx["*"]]])
        generated_name_idx = model.generate(src, dataset.char_to_idx["*"], dataset.char_to_idx["#"], idx_to_char=dataset.idx_to_char, temperature=0.8)
        generated_name = "".join([dataset.idx_to_char[idx.item()] for idx in generated_name_idx[0][1:-1]])
        print("Generated Dog Name:", generated_name)
