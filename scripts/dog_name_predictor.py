import random
import json
from collections import defaultdict, Counter

def load_dog_names(filename):
    """Load dog names from a JSON file."""
    with open(filename, 'r') as file:
        data = json.load(file)  # Assumes the JSON file contains an array directly
    return data

def create_bigrams(names):
    """Create bigrams from a list of names."""
    bigrams = []
    for name in names:
        # Add a special character to denote the start and end of a name
        name = "*" + name + "#"
        for i in range(len(name) - 1):
            bigrams.append((name[i], name[i+1]))
    return bigrams

def build_bigram_model(bigrams):
    """Build a bigram model from a list of bigrams."""
    model = defaultdict(Counter)
    for bigram in bigrams:
        model[bigram[0]][bigram[1]] += 1
    return model

def generate_name(model, max_length=12):
    """Generate a dog name using the bigram model."""
    name = '*'
    while True:
        next_char = random.choices(list(model[name[-1]].keys()), weights=model[name[-1]].values())[0]
        if next_char == '#' or len(name) - 1 == max_length:
            break
        name += next_char
    return name[1:]  # Remove the start character

# Main usage
if __name__ == "__main__":
    filename = 'dog_names.txt'
    names = load_dog_names("./dog_names.json")
    bigrams = create_bigrams(names)
    model = build_bigram_model(bigrams)
    new_name = generate_name(model)
    print("Generated Dog Name:", new_name)
