#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to makegit this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang, Aaron van Beelen, Lotte Bottema
# date: 31 Mar 2017

import argparse
import sys
import pandas as pd

# Start your coding

# import the library you need here
import pickle
from train_models import get_best_features
# End your coding

def to_dataframe(input_file):
    """
    This function reads the input file and makes it into a DataFrame
    """
    # load data as a DataFrame
    data = pd.read_csv(input_file, sep="\s+|\t+|\s+\t+|\t+\s+")

    # remove chromosome, Start, End and Nclone columns to make a "simple" data set
    data.drop(columns=['"Chromosome"', '"Start"', '"End"', '"Nclone"'], inplace=True)

    # transpose DataFrame such that the probes (samples) are rows
    data = data.transpose()

    return data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding
    # suggested steps
    # Step 1: load the model from the model file
    test_data = to_dataframe(args.input_file)
    test_data_features = get_best_features(test_data)
    print(test_data_features.shape)

    # Step 2: apply the model to the input file to do the prediction
    with open(args.model_file, 'rb') as f:
        pkl_model = pickle.load(f)

    predictions = pkl_model.predict(test_data_features)

    # Step 3: write the prediction into the designated output file
    with open(args.output_file, "w") as file:
        file.write('"Sample"\t"Subgroup"\n')
        for sample, subgroup in zip(test_data_features.index, [f'"{x}"' for x in predictions]):
            if sample != '"Array.63"':
                file.write(f"{sample}\t{subgroup}\n")
            else:
                file.write(f"{sample}\t{subgroup}")


    # End your coding

if __name__ == '__main__':
    main()
