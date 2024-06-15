# Dog Name Predictor with Transformer Model

This project is designed to generate dog names using a Transformer model. The script trains a Transformer model on a dataset of dog names and generates new names based on learned patterns.

## Features

- Loads dog names from a CSV file.
- Uses a Transformer model to learn the patterns in dog names.
- Generates new dog names based on the learned patterns.
- Saves and loads tokenization mappings and model state to avoid retraining from scratch.
- Filters out names that are more than one word and longer than eight characters.

## Installation

1. Ensure you have Python 3.10 and the necessary packages installed. You can install the required packages using pip:

    ```bash
    pip install torch
    ```

2. Clone this repository or download the script to your local machine.

## Usage

1. Prepare the dataset:
   - Ensure you have a CSV file named `kaggle-dog-name-frequencies.csv` with the dog names. The script assumes that the names are in the second column of the CSV file.

2. Run the script to train the model and generate a dog name:

    ```bash
    python dog-names-saved-tokens.py
    ```

3. The script will train the model and generate a new dog name based on the learned patterns.

## File Descriptions

- `dog-names-saved-tokens.py`: The main script that trains the Transformer model and generates dog names. This script saves the tokenization mappings and the model state to avoid retraining from scratch.

## Example Output

```bash
Character to Index Mapping: {'#': 0, '*': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 9, 'I': 10, 'J': 11, 'K': 12, 'L': 13, 'M': 14, 'N': 15, 'O': 16, 'P': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26}
No saved model or parameters found at transformer_model.pth and model_params.json
/home/mike/.local/lib/python3.10/site-packages/torch/nn/modules/transformer.py:306: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
Epoch 1, Loss: 2.6400351524353027
Epoch 2, Loss: 2.246021270751953
Epoch 3, Loss: 1.6805933713912964
Epoch 4, Loss: 1.263124704360962
Epoch 5, Loss: 0.5068111419677734
Epoch 6, Loss: 0.47074735164642334
Epoch 7, Loss: 0.1940685361623764
Epoch 8, Loss: 0.5533576011657715
Epoch 9, Loss: 0.1791338175535202
Epoch 10, Loss: 0.19112113118171692
Generated Dog Name: FRODO
```

## Limitations and Recommendations

### Limitations

- **Computational Power**: Training a Transformer model can be computationally intensive, and running this on a home computer with a CPU might be slow and inefficient. For large datasets and more complex models, using a GPU is recommended.

### Recommendations for Cloud-Based Deployment

1. **Use Cloud Services**: Utilize cloud services like AWS, Google Cloud, or Azure, which provide powerful GPUs that can significantly speed up training.
2. **Use Managed ML Services**: Consider using managed machine learning services like Google AI Platform, AWS SageMaker, or Azure Machine Learning, which offer tools for easy deployment, scaling, and management of machine learning models.

To deploy on a cloud service:
1. **Set up a cloud account**: Create an account on a cloud platform of your choice.
2. **Provision a GPU instance**: Create and configure a virtual machine with a GPU.
3. **Transfer the script and dataset**: Upload your script and dataset to the virtual machine.
4. **Run the script**: Execute the script on the virtual machine to train the model and generate names.
5. **Save the model state**: Ensure the model state and tokenization mappings are saved for future use.

## Contributing

Contributions are welcome! If you have any suggestions, please create an issue or submit a pull request.

## License

This project is licensed under the MIT License.

---
