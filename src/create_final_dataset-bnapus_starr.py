# process canola organelle starr dataset

import argparse
import pandas as pd
import numpy as np


def fetch_args():
    parser = argparse.ArgumentParser(description="Canola Organelle STARR Dataset Creation")

    parser.add_argument('-di', '--data_input_path', type=str, default="data/raw/hidra.tsv", help='Path to STARR data')
    parser.add_argument('-do', '--data_output_path', type=str, default="data/processed/bnapus_organelle_starr.csv", help='Formatted dataset output path')

    parser.add_argument('-pa', '--pad_up_to', type=int, default=0, help='Add X to sequences to reach desired length, 0 if leave alone')

    parser.add_argument('-nc', '--num_chunks', type=int, default=70, help='Number of pieces to split the dataset into for train/test/val')
    parser.add_argument('-or', '--organelle', type=str, default="NC_016734.1", help='NC_016734.1 for chloroplast, NC_008285.1 for mitochondria')

    args = parser.parse_args()

    return args


def trim(chunks):  
    '''
    Takes in list of dfs split evenly into chunks
    Returns list where dfs have no overlapping sequences
    note: this is hardcoded to work when the step is 5 and sequence length is 145
    '''
  
    chunks[0] = chunks[0][:-14]
    for i in range(1, len(chunks)-1):
      chunks[i] = chunks[i][14:-14]
    chunks[-1] = chunks[-1][14:]

    return chunks


def main():
    args = fetch_args()  # get args

    df = pd.read_csv(args.data_input_path, sep="\t", header=None)
    df.columns = ["ref", 
                  "start_coord", 
                  "end_coord", 
                  "sequence", 
                  "control_raw_coverage", 
                  "treatment_raw_coverage",
                  "control_norm_coverage",
                  "treatment_norm_coverage",
                  "target"]

    df[df.ref == args.organelle]  # restricts to organelle we want

    # get rid of weirdo rows
    odds = [s for s in list(set("".join(df.sequence))) if s not in ["A", "T", "C", "G"]]
    if len(odds) > 0:
      mask = df.sequence.str.contains("|".join(odds))  # true if contains weird characters, false if contains only ATCG
      df = df[np.logical_not(mask)]  # keep only rows without weird characters

    # splits dataset into n chunks to create even train/test/val split
    chunks = trim(np.array_split(df, args.num_chunks))  

    training = []
    validation = []
    test = []

    for i in range(len(chunks)):   # divides each of n non-overlapping chunks into train/test/val, append to train/test/val lists
        idx = int((chunks[i].shape[0] - 29*2)*0.1) + 14  # index of 10% mark after trimming
        
        trimmed = trim([chunks[i][:idx], chunks[i][idx:-idx], chunks[i][-idx:]])  # get rid of all overlapping sequences
        
        test.append(trimmed[0])
        training.append(trimmed[1])
        validation.append(trimmed[2])

    train = pd.concat(training)
    train["set"] = "train"
    val = pd.concat(validation)
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
