featselection
=========

This project provides a set filter methods for feature selection applied to text classification.

Currently the following methods are available:

- ALOFT - At Least One FeaTure `[1] <https://www.sciencedirect.com/science/article/pii/S0957417412007063>`_
- MFD - Maximum f Features per Document `[2] <https://www.sciencedirect.com/science/article/pii/S0957417414006344>`_
- MFDR - Maximum f Features per Document-Reduced `[2] <https://www.sciencedirect.com/science/article/pii/S0957417414006344>`_
- cMFDR - Class-dependent Maximum f Features per Document-Reduced `[3] <https://ieeexplore.ieee.org/abstract/document/7727649>`_
- AFSA - Automatic Features Subsets Analyzer `[4] <https://ieeexplore.ieee.org/abstract/document/7839596>`_

============
Installation
============
The package can be installed using pip:

``pip install featselection``

=============
Dependencies
=============
The code is tested to work with Python 3.6. The dependency requirements are: 

* numpy
* scipy
* pandas
* scikit-learn

These dependencies are automatically installed using the pip command above.

=========
Examples
=========

In this example, we show the use MFD.

.. code-block:: python3

    import numpy as np

    from sklearn.metrics import accuracy_score
    from sklearn.feature_selection import chi2
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_extraction.text import CountVectorizer

    from filters import MFD


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
        my_filter = MFD(10, chi2)
        X_train = my_filter.fit_transform(vectors[train_index], newsgroups.target[train_index])
        clf = MultinomialNB()
        clf.fit(X_train, newsgroups.target[train_index])

        # Test
        X_test = my_filter.transform(vectors[test_index])
        predicted = clf.predict(X_test)

        # Evaluate
        accuracy_results.append(accuracy_score(newsgroups.target[test_index], predicted))

    # Output averaged accuracy
    print('Mean accuracy = {0} ({1})'.format(np.mean(accuracy_results), np.std(accuracy_results)))

==========
References
==========



`[1] <https://www.sciencedirect.com/science/article/pii/S0957417412007063>`_ Pinheiro, Roberto HW, et al. "A global-ranking local feature selection method for text categorization." Expert Systems with Applications 39.17 (2012): 12851-12857.

`[2] <https://www.sciencedirect.com/science/article/pii/S0957417414006344>`_ Pinheiro, Roberto HW, et al. "Data-driven global-ranking local feature selection methods for text categorization." Expert Systems with Applications 42.4 (2015): 1941-1949.

`[3] <https://ieeexplore.ieee.org/abstract/document/7727649>`_ Fragoso, Rogério CP, et al. "Class-dependent feature selection algorithm for text categorization." 2016 International Joint Conference on Neural Networks (IJCNN). IEEE, 2016.

`[4] <https://ieeexplore.ieee.org/abstract/document/7839596>`_ Fragoso, Rogério CP, et al. "A method for automatic determination of the feature vector size for text categorization." 2016 5th Brazilian Conference on Intelligent Systems (BRACIS). IEEE, 2016.
