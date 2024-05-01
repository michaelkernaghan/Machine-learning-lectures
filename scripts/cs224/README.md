# Python Co-Occurrence and Word Embedding Visualization Tool

This Python program provides tools for analyzing text corpora through co-occurrence matrices and word embeddings, utilizing libraries such as NLTK, Gensim, and Scikit-Learn. It includes functions to read and preprocess text, compute co-occurrence matrices, reduce dimensions with SVD, and plot word embeddings.

## Prerequisites

Ensure you have Python 3.8 or higher installed. This program uses several advanced Python libraries for natural language processing and matrix operations.

### Required Python Version

- Python 3.8 or higher

### Libraries Used

- NLTK
- Gensim
- NumPy
- SciPy
- Matplotlib
- Scikit-Learn

## Installation

1. **Clone the repository:**

```bash
git clone <repository-url>
cd <repository-directory>
```

2. **Set up a Python virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install the required packages:**

```bash
pip install -r requirements.txt
```
If there is an error 
```bash
ImportError: cannot import name 'triu' from 'scipy.linalg'
``` 
then you need to 
```bash
pip install "scipy<1.13"
```
## Usage

Run the script from the command line:

```bash
python <script-name>.py
```

This will execute the predefined processes such as reading the corpus, computing the co-occurrence matrix, reducing dimensions, and plotting embeddings.

## Functions

- `read_corpus()`: Reads and processes files from a specified category using NLTK's Reuters corpus.
- `distinct_words()`: Identifies unique words in the corpus.
- `compute_co_occurrence_matrix()`: Generates a co-occurrence matrix from the corpus.
- `reduce_to_k_dim()`: Reduces the dimensionality of the co-occurrence matrix using SVD.
- `plot_embeddings()`: Plots the reduced word embeddings.

### Example Output

The script will output tests confirming the correctness of its functions and save plots of word embeddings to the specified directory.

## Upgrading Python

If your Python version is below 3.8, please upgrade by downloading the latest Python version from the official [Python website](https://www.python.org/downloads/).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK for providing the Reuters corpus.
- Gensim for the pretrained word vectors.
- Scikit-Learn for machine learning tools.
```
