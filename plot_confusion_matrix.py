import argparse

import matplotlib.pyplot as plt
import numpy as np
import opfython.math.distance as d
import seaborn as sbn


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    parser = argparse.ArgumentParser(usage='Plots the confusion matrix.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['arcene', 'basehock', 'caltech101', 'coil20',
                                                                       'isolet', 'lung', 'madelon', 'mpeg7', 'mpeg7_BAS',
                                                                       'mpeg7_FOURIER', 'mushrooms', 'ntl-commercial',
                                                                       'ntl-industrial', 'orl', 'pcmac', 'phishing',
                                                                       'segment', 'semeion', 'sonar', 'spambase',
                                                                       'tor-nontor', 'vehicle', 'wine'])

    parser.add_argument('seed', help='Deterministic seed', type=int)

    parser.add_argument('-idx', help='Distance metric index', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    seed = args.seed
    idx = args.idx
    input_file = f'output/{dataset}_{seed}_matrix.npy'
    output_file = f'output/{dataset}_{seed}_report.txt'

    # Loading the input file
    c_matrix = np.load(input_file)

    # Iterates through every distance metric
    for i, key in enumerate(d.DISTANCES.keys()):
        # Checks if its corresponding distance
        if i == idx:
            # Defining a seaborn's heatmap
            sbn.heatmap(c_matrix[idx], cmap='RdPu', annot=True)

            # Defining the title and labels
            plt.title(f'Dataset: {dataset} | Distance: {key}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

            # Showing the plot
            plt.show()
