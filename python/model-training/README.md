# Dog Name Generator with GPT

This project uses a Generative Pre-trained Transformer (GPT) model to generate dog names. The GPT model is trained on a dataset of dog names and can generate new, creative names based on the patterns it has learned.

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pandas

## Installation

1. copy this Python script:
    ```sh
    git clone https://github.com/michaelkernaghan/Machine-learning-lectures/blob/main/python/model-training/dog-name-GPT.py
    ```

2. Install the required packages:
    ```sh
    pip install torch numpy pandas
    ```

## Dataset

The dataset used for training the model is a CSV file containing dog names. The file should be named `kaggle-dog-name-frequencies.csv` and should be placed in the root directory of this project.

## Training the Model

To train the GPT model, run the following command:

```sh
python dog-name-GPT.py
```

The script will:
1. Load the dog names from the CSV file.
2. Tokenize the names and save the tokenization mappings.
3. Train the GPT model on the dataset.
4. Save the trained model and the tokenization mappings.

### Tokenization

The tokenization process converts the dog names into a format that can be fed into the GPT model. The tokenization mappings are saved to a file named `tokenization.json` to avoid redundant processing in future runs.

### Model Training

The model is trained for a specified number of epochs. If a saved model with the correct vocabulary size exists, it will be loaded; otherwise, a new model will be trained.

### Generating Names

After training, the model can generate new dog names based on the patterns it has learned. The generated names will be displayed in the console.

## Using Cloud Resources

Training deep learning models like GPT can be resource-intensive. It is recommended to use cloud-based resources for faster training and better performance. Here are a few options:

- **Google Colab**: Provides free access to GPUs. You can upload your project to Google Colab and train the model there.
- **AWS SageMaker**: A managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning models quickly.
- **Microsoft Azure ML**: An end-to-end cloud service for building, training, and deploying machine learning models.

### Running on Google Colab

1. Upload your project files to Google Drive.
2. Open a new notebook in Google Colab.
3. Mount your Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
4. Navigate to your project directory:
    ```python
    %cd /content/drive/MyDrive/path_to_your_project/
    ```
5. Run the training script:
    ```python
    !python dog-name-GPT.py
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bugs, improvements, or feature requests.

## License

This project is licensed under the MIT License.
```