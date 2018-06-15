#!/usr/bin/env python

import pandas as pd
from os.path import basename, splitext
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inputfile', type=str, help='input file of raw data')
parser.add_argument('outputdir', type=str, help='output directory')
parser.add_argument('colnames', type=str, help='txt file containing column names of data set')
parser.add_argument('--response', type=str)
parser.add_argument('--drop', type=str)
args = parser.parse_args()

if __name__ == "__main__":

    with open(args.colnames, 'r') as f:
        colnames = f.read().split(' ')

    if args.drop:
        drop = args.drop.split(' ')
    else:
        drop = [None]

    keep = list(set(colnames).difference(set(drop)))

    data_file = pd.read_table(args.inputfile,
                              engine='python',
                              sep="\s+|\t+|\s+\t+|\t+\s+",
                              names=colnames, 
                              header=None,
                              usecols=keep)

    data_file = data_file.dropna()

    cols = list(data_file)
    cols.insert(0, cols.pop(cols.index(args.response)))
    data_file = data_file.ix[:, cols]

    filename = splitext(basename(args.inputfile))[0]
    data_file.to_csv(args.outputdir + filename + '.csv', index=False)
    