# Word Vectors Demo

## Overview

This script demonstrates the use of pre-trained GloVe word embeddings for various natural language processing (NLP) tasks. It loads the GloVe embeddings, converts them to a format compatible with Gensim's Word2Vec, and performs several word vector operations including finding similar words, analogies, identifying non-matching words, and visualizing word embeddings using Principal Component Analysis (PCA).

## Purpose

The purpose of this script is to provide an example of how to utilize GloVe word embeddings for NLP tasks. It showcases the capabilities of word embeddings in understanding word relationships and visualizing high-dimensional data.

## Functionality

1. **Loading and Validating GloVe Word Embeddings:**
   - The script begins by validating and loading the GloVe word embeddings from a specified file.

2. **Finding Similar Words:**
   - The script uses the word embeddings to find and print words that are most similar to specified words.

3. **Performing Word Analogies:**
   - The script demonstrates how to perform vector arithmetic to find analogies (e.g., "woman" + "king" - "man" = "queen").

4. **Identifying Non-Matching Words:**
   - The script identifies the word that does not match in a given list of words.

5. **Visualizing Word Embeddings:**
   - The script uses PCA to reduce the dimensionality of the word vectors and visualizes them in a scatter plot.

## Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn
- Gensim

## Setup

1. **Ensure you have Python 3.x installed on your system.**

2. **Install the required libraries:**
    ```sh
    pip install numpy matplotlib scikit-learn gensim
    ```

3. **Download the GloVe word embeddings:**
   - Download the `glove.6B.50d.txt` file from the [GloVe website](https://nlp.stanford.edu/projects/glove/) and place it in your working directory.

## Running the Script

1. **Run the script:**
    ```sh
    python3 gensim-word-vectors-demo.py
    ```

2. **Expected Output:**
   - The script will print the most similar words to specified words, perform word analogies, identify non-matching words, and display scatter plots of word embeddings.

## Script Details

### Loading and Validating GloVe Word Embeddings

The script begins by setting up the path for the GloVe file and includes a validation function to ensure the file contains valid numeric data.

### Finding Similar Words

The script demonstrates finding similar words using the `most_similar` method of the `KeyedVectors` class.

### Performing Word Analogies

The script includes a function `analogy` to perform vector arithmetic for word analogies, showcasing the capability of word embeddings to understand word relationships.

### Identifying Non-Matching Words

Using the `doesnt_match` method, the script identifies the word that does not belong in a given list.

### Visualizing Word Embeddings

The script includes a function `display_pca_scatterplot` to visualize word embeddings using PCA, making it easier to understand the relationships between words in a lower-dimensional space.

## Example Output

- Similar words to 'obama'
- Similar words to 'banana'
- Opposite of 'banana'
- Result of 'woman' + 'king' - 'man'
- Various analogies such as 'japan' -> 'japanese' and 'australia'
- Identification of the non-matching word in a list
- PCA scatter plots of word embeddings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
