# process arabidopsis bac starr dataset 

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
import time
import random


def fetch_args():
    parser = argparse.ArgumentParser(description="Arabidopsis BAC STARR Dataset Creation")

    parser.add_argument('-di', '--data_input_path', type=str, default="data/raw/athal_bac.intro.tsv", help='Path to BAC data')
    parser.add_argument('-do', '--data_output_path', type=str, default="data/processed/athaliana_bac_starr.csv", help='Formatted dataset output path')

    parser.add_argument('-pa', '--pad_up_to', type=int, default=0, help='Add X to sequences to reach desired length, 0 if leave alone')
    parser.add_argument('-nc', '--num_chunks', type=int, default=10, help='Number of pieces to split the dataset into for train/test/val')

    args = parser.parse_args()

    return args


def trim(chunks): 
    '''
    Takes in list of dfs split evenly into chunks
    Returns list where dfs have no overlapping sequences
    note: this is hardcoded to work when the step is 17 and sequence length is 153
    '''

    num = int((153/17-1)/2)  # how does this get calculated?
    chunks[0] = chunks[0][:-num]  # knock off the last <num> rows from first
    for i in range(1, len(chunks)-1):
        chunks[i] = chunks[i][num:-num]
    chunks[-1] = chunks[-1][num:]  # knock off the first <num> rows from last
    return chunks


def main():
    args = fetch_args()  # get args

    df = pd.read_csv(args.data_input_path, sep="\t", header=None)
    df.columns = ["ref", 
                  "start_coord", 
                  "end_coord", 
                  "sequence", 
                  "raw_control_coverage", 
                  "raw_treatment_coverage", 
                  "norm_control_coverage", 
                  "norm_treatment_coverage"]

    # get rid of rows with non-standard characters
    odds = [s for s in list(set("".join(df.sequence))) if s not in ["A", "T", "C", "G"]]
    if len(odds) > 0:
      mask = df.sequence.str.contains("|".join(odds))  # true if contains weird characters, false if contains only ATCG
      df = df[np.logical_not(mask)]  # keep only rows without weird characters 

    # create target column
    df["target"] = np.log2(df.norm_control_coverage/df.norm_treatment_coverage)

    # create train/test/val split
    chunks = trim(np.array_split(df, args.num_chunks))

    train = []
    val = []
    test = []

    for i in range(len(chunks)):   # divides each of n chunks into train/test/val, append to train/test/val lists
        idx = int((chunks[i].shape[0] - 153/17*2)*0.1) + 4  # index of 10% mark after trimming
        
        trimmed = trim([chunks[i][:idx], chunks[i][idx:-idx], chunks[i][-idx:]])  # get rid of all overlapping sequences
        
        test.append(trimmed[0])
        train.append(trimmed[1])
        val.append(trimmed[2])

    train = pd.concat(train)
    train["set"] = "train"
    val = pd.concat(val)
    val["set"] = "val"
    test = pd.concat(test)
    test["set"] = "test"

    # bring sequences up to desired length using Xs
    if args.pad_up_to > 0:
      difference = args.pad_up_to - len(df.sequence.tolist()[0])
      assert difference >= 0  # only contains functionality to pad upwards
      df["sequence"] += "X"*difference

    df = pd.concat([train, val, test]).sort_index()
    df = df[["ref", "start_coord", "end_coord", "sequence", "target", "set"]]

    df.to_csv(args.data_output_path, index=False)


main()
