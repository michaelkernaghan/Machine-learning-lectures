# Dog Name Generator Using Bigrams

This Python script generates dog names using a bigram model. It reads a list of existing dog names from a JSON file, constructs bigrams from these names, and builds a probabilistic model. The model can then generate new dog names based on the patterns learned from the input data.

## Features

- Loads dog names from a JSON file.
- Creates bigrams from the names to capture the transition probabilities between characters.
- Generates new names using the probabilistic model of bigrams.

## Installation

To run this script, you will need Python installed on your machine. The script is compatible with Python 3. You will also need to ensure that you have the `json` module, which is included in Python's standard library.

### Dependencies

- Python 3
- `json` module (standard library)
- `random` module (standard library)
- `collections` module (standard library)

## Usage

1. **Prepare Your Data:**
   - Ensure you have a JSON file named `dog_names.json` that contains an array of dog names. Here is an example format for the JSON file:
   ```json
   ["Bella", "Lucy", "Max", "Charlie"]
   ```

2. **Running the Script:**
   - Clone the repository or download the script to your local machine.
   - Place the JSON file in the same directory as the script, or modify the script to point to the location of your JSON file.
   - Run the script using Python:
   ```bash
   python dog_name_generator.py
   ```

3. **Generating Names:**
   - The script will automatically load the dog names, create a bigram model, and generate a new dog name.
   - The generated name will be printed to the console.

## How It Works

- **Loading Names:** The script starts by loading dog names from a JSON file.
- **Creating Bigrams:** It then processes each name to create bigrams, which are pairs of consecutive characters in the names.
- **Building the Model:** A probabilistic model is built from these bigrams, where the probability of each character following another is calculated.
- **Generating Names:** Finally, the script uses this model to generate a new dog name based on the probabilities of character transitions.

## Contributing

Contributions to this project are welcome. You can contribute by improving the algorithm, adding features, or refining the documentation.

## License

This project is open-sourced under the MIT license.
