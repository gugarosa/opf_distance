import numpy as np
import argparse

import opfython.math.distance as d
import opfython.math.general as g
from opfython.models import SupervisedOPF

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Classifies data using Optimum-Path Forest.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['boat'])

    parser.add_argument('-tr_split', help='Training set percentage', type=float, default=0.5)

    parser.add_argument('-seed', help='Deterministic seed', type=int, default=0)

    parser.add_argument('--normalize', help='Whether data should be normalized or not', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    input_file = f'data/{args.dataset}.txt'
    split = args.tr_split
    seed = args.seed
    normalize = args.normalize

    # Loads the training and testing sets along their indexes
    X_train, Y_train, X_test, Y_test = l.load_split_dataset(
        input_file, train_split=split, normalize=normalize, random_state=seed)

    # Gathers the amount of distance metrics and classes
    n_distances = len(d.DISTANCES)
    n_classes = max(Y_train)

    # Creates an array to save the output confusion matrices
    c_matrix = np.zeros((n_distances, n_classes, n_classes))

    # Iterates through every distance
    for i, key in enumerate(d.DISTANCES.keys()):
        # Creates a SupervisedOPF with the iterated distance
        opf = SupervisedOPF(distance=key)

        # Fits training data into the classifier
        opf.fit(X_train, Y_train)

        # Predicts new data
        preds = opf.predict(X_test)

        # Calculating accuracy
        c_matrix[i] = g.confusion_matrix(Y_test, preds)

    # Saving confusion matrix in a pickle file
    np.save(f'{args.dataset}_seed.pkl', c_matrix)
