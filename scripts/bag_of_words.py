import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the documents with different themes for classification
documents = [
    "I see many people die because they judge that life is not worth living. I see others paradoxically gettin killed for the ideas or illusions that give them a reason for living (what is called a reason for living is also an excellent reason for dying). I therefore conclude that the meaning of life is the most urgent of questions.",
    "Laws are partly formed for the sake of good men, in order to instruct them how they may live on friendly terms with one another, and partly for the sake of those who refuse to be instructed, whose spirit cannot be subdued, or softened, or hindered from plunging into evil.",
    "I started thinking about little kids putting a cylindrical peg through a circular hole, and how they do it over and over again for months when they figure it out, and how basketball was basically just a slightly more aerobic version of that same exercise."
]

# Corresponding labels for each document representing their categories
labels = ['philosophy', 'law', 'psychology']

# Initialize the CountVectorizer to convert text documents into a matrix of token counts
vectorizer = CountVectorizer()
# Fit the model on the documents and transform the text data into a sparse matrix of word counts
bow = vectorizer.fit_transform(documents)

# Display the vocabulary to show how words are indexed
print("Vocabulary:")
print(vectorizer.vocabulary_)
print("\nBag of Words (Sparse Matrix):")
print(bow.toarray())

# Split data into training and test sets for model evaluation
X_train, X_test, y_train, y_test = train_test_split(bow, labels, test_size=0.66, random_state=42)
print("\nTraining and testing set sizes:", X_train.shape, X_test.shape)

# Train a Naive Bayes classifier, suitable for classification with discrete features like word counts
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predict the categories of the test set documents
predictions = classifier.predict(X_test)

# Evaluate the classifier's accuracy
accuracy = accuracy_score(y_test, predictions)

# Display predictions and accuracy
print("\nPredicted categories:")
print(predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
