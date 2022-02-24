#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import os
import argparse
from scipy.stats import ttest_rel
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

# set the random state for reproducibility
import numpy as np

np.random.seed(401)

CLF_DICT = {
    0: SGDClassifier,
    1: GaussianNB,
    2: RandomForestClassifier,
    3: MLPClassifier,
    4: AdaBoostClassifier
}


def accuracy(c):
    """ Compute accuracy given Numpy array confusion matrix C. Returns a floating point value. """
    n, _ = c.shape
    return sum([c[i, i] for i in range(n)]) / sum([c[i, j] for i in range(n) for j in range(n)])


def recall(c):
    """ Compute recall given Numpy array confusion matrix C. Returns a list of floating point values. """
    n, _ = c.shape
    return [(c[k, k] / sum([c[k, j] for j in range(n)])) for k in range(n)]


def precision(c):
    """ Compute precision given Numpy array confusion matrix C. Returns a list of floating point values """
    n, _ = c.shape
    return [(c[k, k] / sum([c[i, k] for i in range(n)])) for k in range(n)]


def class31(output_dir, X_train, X_test, y_train, y_test):
    """ This function performs experiment 3.1
        Parameters
           output_dir: path of directory to write output to
        Returns:
           i: int, the index of the supposed best classifier
    """
    print('Section 3.1')

    # 1. SGDClassifier: support vector machine with a linear kernel.
    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)

    # 2. GaussianNB: a Gaussian naive Bayes classifier.
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # 3. RandomForestClassifier: with a maximum depth of 5, and 10 estimators.
    rf = RandomForestClassifier(max_depth=5, n_estimators=10)
    rf.fit(X_train, y_train)

    # 4. MLPClassifier: A feed-forward neural network, with α = 0.05.
    mlp = MLPClassifier(alpha=0.05)
    mlp.fit(X_train, y_train)

    # 5. AdaBoostClassifier: with the default hyper-parameters.
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)

    scores = []
    with open(f"{output_dir}/a1_3.1.txt", "w") as out_f:
        # For each classifier, compute results and write the following output:
        for clf in [sgd, gnb, rf, mlp, ada]:
            c = confusion_matrix(y_test, clf.predict(X_test))
            acc = accuracy(c)
            scores.append(acc)
            out_f.write(f'Results for {str(clf)}:\n')  # Classifier name
            out_f.write(f'\tAccuracy: {acc:.4f}\n')
            out_f.write(f'\tRecall: {[round(item, 4) for item in recall(c)]}\n')
            out_f.write(f'\tPrecision: {[round(item, 4) for item in precision(c)]}\n')
            out_f.write(f'\tConfusion Matrix: \n{c}\n\n')
    return int(np.argmax(scores))


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    """ This function performs experiment 3.2
    Parameters:
       output_dir: path of directory to write output to
       iBest: int, the index of the supposed best classifier (from task 3.1)
    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    """
    print('Section 3.2')
    clf = CLF_DICT[iBest]()
    with open(f"{output_dir}/a1_3.2.txt", "w") as out_f:
        # For each number of training examples, compute results and write
        for num_train in map(int, [1e4, 5e4, 1e5, 15e4, 2e5]):  # 1K, 5K, 10K, 15K, and 20K
            clf.fit(X_train[:num_train], y_train[:num_train])
            c = confusion_matrix(y_test, clf.predict(X_test))
            acc = accuracy(c)
            out_f.write(f'{num_train}: {acc:.4f}\n')

    return X_train[:1000], y_train[:1000]


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    """ This function performs experiment 3.3
        Parameters:
           output_dir: path of directory to write output to
           i: int, the index of the supposed best classifier (from task 3.1)
           X_1k: numPy array, just 1K rows of X_train (from task 3.2)
           y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    """
    print('Section 3.3')
    with open(f"{output_dir}/a1_3.3.txt", "w") as out_f:
        # Prepare the variables with corresponding names, then uncomment this, so it writes them to outf.

        """ 1. For the 32k training set and each number of features k = {5, 50}, 
            find the best k features according to this approach. 
            Write the associated p-values to a1 3.3.txt using the format strings provided.
        """
        for k in [5, 50]:
            # For each number of features k_feat, write the p-values for that number of features:
            selector = SelectKBest(k=k)
            selector.fit(X_train, y_train)

            pp = selector.pvalues_[np.argpartition(selector.scores_, -k)[-k:]]
            out_f.write(f'{k} p-values: {[format(pval) for pval in pp]}\n')

        """ 2. Train the best classifier from section 3.1 for each of the 1K training set and the 32K training set,
        using only the best k = 5 features. Write the accuracies on the full test set of both classifiers to
        a1 3.3.txt using the format strings provided.
        """
        selector = SelectKBest(k=5)
        X_new = selector.fit_transform(X_train, y_train)
        X_test_new = selector.transform(X_test)

        clf = CLF_DICT[i]()
        clf.fit(X_new[:1000], y_train[:1000])
        c = confusion_matrix(y_test, clf.predict(X_test_new))
        accuracy_1k = accuracy(c)
        out_f.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')

        clf = CLF_DICT[i]()
        clf.fit(X_new, y_train)
        c = confusion_matrix(y_test, clf.predict(X_test_new))
        accuracy_full = accuracy(c)
        out_f.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')

        """ 3. Extract the indices of the top k = 5 features using the 1K training set and take the intersection with
           the k = 5 features using the 32K training set. Write using the format strings provided.
        """
        top_1k_selector = SelectKBest(k=5)
        top_1k_selector.fit(X_train[:1000], y_train[:1000])
        top_1k_index = set(np.argpartition(top_1k_selector.scores_, -5)[-5:])

        top_all_selector = SelectKBest(k=5)
        top_all_selector.fit(X_train, y_train)
        top_all_index = set(np.argpartition(top_all_selector.scores_, -5)[-5:])

        feature_intersection = top_1k_index.intersection(top_all_index)
        out_f.write(f'Chosen feature intersection: {feature_intersection}\n')

        """
        4. Format the top k = 5 feature indices extracted from the 32K training set to file using the format 
        string provided.
        """
        out_f.write(f'Top-5 at higher: {top_all_index}\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    """ This function performs experiment 3.4
    Parameters
       output_dir: path of directory to write output to
       i: int, the index of the supposed best classifier (from task 3.1)
    """
    print('Section 3.4')

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.

        sgd = SGDClassifier()
        gnb = GaussianNB()
        rf = RandomForestClassifier(max_depth=5, n_estimators=10)
        mlp = MLPClassifier(alpha=0.05)
        ada = AdaBoostClassifier()

        kfold_accuracies = []
        for clf in [sgd, gnb, rf, mlp, ada]:
            kf = KFold(n_splits=5, random_state=None, shuffle=False)
            kfold_accs = []
            for train_index, test_index in kf.split(X_train):
                X_train_s, X_test_s = X_train[train_index], X_train[test_index]
                y_train_s, y_test_s = y_train[train_index], y_train[test_index]
                clf.fit(X_train_s, y_train_s)
                kfold_accs.append(accuracy(confusion_matrix(y_test_s, clf.predict(X_test_s))))
            kfold_accuracies.append(int(np.mean(kfold_accs)))
        outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')


if __name__ == "__main__":
    """
    python a1_classify.py -i feats.npz -o .
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument("-o", "--output_dir",
                        help="The directory to write a1_3.X.txt files to.",
                        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    """ Load data and split into train and test. """
    with np.load(args.input) as npz_f:
        feat = npz_f["arr_0"]
    x, y = feat[:, :-1], feat[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    """ Complete each classification experiment, in sequence. """
    i_best = class31(args.output_dir, X_train, X_test, y_train, y_test)
    x_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, i_best)
    class33(args.output_dir, X_train, X_test, y_train, y_test, i_best, x_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, i_best)
