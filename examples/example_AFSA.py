import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

from featselection.filter import AFSA, MFD

# Load data
cats = ['comp.windows.x', 'rec.sport.baseball', 'sci.med', 'soc.religion.christian', 'talk.politics.misc']
newsgroups = fetch_20newsgroups(categories=cats)

# Pre-processing: Transform texts to Bag-of-Words and remove stopwords
vectorizer = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(newsgroups.data)

# 10-fold stratified cross validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracy_results = []

for train_index, test_index in skf.split(vectors, newsgroups.target):
    # Train
    my_filter = AFSA(10, chi2, MFD, MultinomialNB, accuracy_score)
    X_train = my_filter.fit_transform(vectors[train_index], newsgroups.target[train_index])
    clf = MultinomialNB()
    clf.fit(X_train, newsgroups.target[train_index])

    # Test
    X_test = my_filter.transform(vectors[test_index])
    predicted = clf.predict(X_test)

    # Evaluate
    accuracy_results.append(accuracy_score(newsgroups.target[test_index], predicted))

# Output averaged accuracy
print('Mean accuracy = {:.4f} ({:.4f})'.format(np.mean(accuracy_results), np.std(accuracy_results)))
