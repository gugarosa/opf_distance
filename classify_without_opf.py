import argparse
import pickle

import numpy as np
import opfython.math.general as g
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Classifies data using Scikit-Learn classifiers.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['arcene', 'basehock', 'caltech101', 'coil20',
                                                                       'isolet', 'lung', 'madelon', 'mpeg7', 'mpeg7_BAS',
                                                                       'mpeg7_FOURIER', 'mushrooms', 'ntl-commercial',
                                                                       'ntl-industrial', 'orl', 'pcmac', 'phishing',
                                                                       'segment', 'semeion', 'sonar', 'spambase',
                                                                       'vehicle', 'wine'])

    parser.add_argument('-tr_split', help='Training set percentage', type=float, default=0.5)

    parser.add_argument('-seed', help='Deterministic seed', type=int, default=0)

    parser.add_argument('--normalize', help='Whether data should be normalized or not', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    input_file = f'data/{dataset}.txt'
    input_sim = f'data/{dataset}_sim.txt'
    split = args.tr_split
    seed = args.seed
    normalize = args.normalize

    # Loads the training and testing sets along their indexes
    X_train, Y_train, X_test, Y_test = l.load_split_dataset(
        input_file, train_split=split, normalize=normalize, random_state=seed)

    # Instantiates the classifier
    clf = DecisionTreeClassifier()
    # clf = LogisticRegression()
    # clf = SVC()

    # Fits training data into the classifier
    clf.fit(X_train, Y_train)

    # Predicts new data
    preds = clf.predict(X_test)

    # Calculates the confusion matrix
    c_matrix = g.confusion_matrix(Y_test, preds)

    # Calculates the classification report
    report = classification_report(Y_test, preds, output_dict=True)

    # Saves confusion matrix in a .npy file
    np.save(f'outputs/{dataset}_{seed}_matrix', c_matrix)

    # Opens file to further save
    with open(f'outputs/{dataset}_{seed}_report.pkl', 'wb') as f:
        # Saves report to a .pkl file
        pickle.dump(report, f)
