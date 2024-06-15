import numpy as np

# Importing interactive tools for Matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Import PCA for dimensionality reduction
from sklearn.decomposition import PCA

# Import utilities and models for word embeddings from Gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Setting up paths for GloVe and converting it to Word2Vec format
glove_file = datapath('/home/mike/Machine-learning-lectures/data/word2vec/glove6b50d.txt')
word2vec_glove_file = get_tmpfile("glove.6B.50d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)

# Load the model
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)

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
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [word for word in model.vocab]
        
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
