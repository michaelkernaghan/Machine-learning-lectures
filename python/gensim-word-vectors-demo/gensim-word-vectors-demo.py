import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

# This script loads pre-trained GloVe word embeddings, converts them to Word2Vec format, 
# and performs various word vector operations including finding similar words, analogies, 
# identifying non-matching words, and visualizing word embeddings using PCA.

# Setting up paths for GloVe
glove_file = "glove.6B.50d.txt"

# Validate the GloVe file
def validate_glove_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                print(f"Invalid line: {line}")
                return False
            try:
                # Attempt to convert the vector components to float
                _ = [float(x) for x in parts[1:]]
            except ValueError:
                print(f"Non-numeric data found: {parts[1:]}")
                return False
    return True

if not validate_glove_file(glove_file):
    raise ValueError("The GloVe file is invalid or contains non-numeric data.")

# Load the GloVe model directly
model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

# Print similar words and their similarity scores
print("Similar words to 'obama':", model.most_similar('obama'))
print("Similar words to 'banana':", model.most_similar('banana'))
print("Opposite of 'banana':", model.most_similar(negative='banana'))

# Perform a vector analogy
result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print("Result of 'woman' + 'king' - 'man':", result[0])

# Function to find an analogous word
def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    print("Analogy - {}: {} -> {}: {}".format(x1, x2, y1, result[0]))
    return result[0][0]

# Running various analogies
analogy('japan', 'japanese', 'australia')
analogy('australia', 'beer', 'france')
analogy('obama', 'clinton', 'reagan')
analogy('tall', 'tallest', 'long')
analogy('good', 'fantastic', 'bad')

# Find which word does not match
non_matching = model.doesnt_match("breakfast cereal dinner lunch".split())
print("Word that doesn't match:", non_matching)

# Display PCA scatterplot of words
def display_pca_scatterplot(model, words=None, sample=0):
    if words is None:
        if sample > 0:
            words = np.random.choice(list(model.key_to_index.keys()), sample)
        else:
            words = list(model.key_to_index.keys())
        
    word_vectors = np.array([model[w] for w in words])
    twodim = PCA().fit_transform(word_vectors)[:, :2]
    
    plt.figure(figsize=(6, 6))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, twodim):
        plt.text(x + 0.05, y + 0.05, word)

# Call the function with specified words
display_pca_scatterplot(model, [
    'coffee', 'tea', 'beer', 'wine', 'brandy', 'rum', 'champagne', 'water',
    'spaghetti', 'borscht', 'hamburger', 'pizza', 'falafel', 'sushi', 'meatballs',
    'dog', 'horse', 'cat', 'monkey', 'parrot', 'koala', 'lizard',
    'frog', 'toad', 'monkey', 'ape', 'kangaroo', 'wombat', 'wolf',
    'france', 'germany', 'hungary', 'luxembourg', 'australia', 'fiji', 'china',
    'homework', 'assignment', 'problem', 'exam', 'test', 'class',
    'school', 'college', 'university', 'institute'])

# Call the function with a random sample
display_pca_scatterplot(model, sample=300)
plt.show()
