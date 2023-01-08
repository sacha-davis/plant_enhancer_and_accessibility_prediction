# process unannotated arabidopsis genome starr dataset 

import argparse
import numpy as np
import pandas as pd
import random


def fetch_args():
    parser = argparse.ArgumentParser(description="Unannotated Arabidopsis Genome STARR Dataset Creation")

    parser.add_argument('-di', '--data_input_path', type=str, default="data/raw/athal_starr_hidra.tsv", help='Path to STARR (or iSTARR) data')
    parser.add_argument('-do', '--data_output_path', type=str, default="data/processed/athaliana_genome_starr.csv", help='Formatted dataset output path')

    parser.add_argument('-sa', '--sample', type=int, default=1, help='Sample every n rows')
    parser.add_argument('-pa', '--pad_up_to', type=int, default=0, help='Add X to sequences to reach desired length, 0 if leave alone')

    parser.add_argument('-sc', '--control_threshold', type=int, default=30, help='Raw control coverage threshold')
    parser.add_argument('-st', '--treatment_threshold', type=int, default=5, help='Raw treatment coverage threshold')

    args = parser.parse_args()

    return args


def main():
    args = fetch_args()  # get args

    df = pd.read_csv(args.data_input_path, sep="\t", skiprows=lambda x: x % args.sample, header=None)
    df.columns = ["ref", 
                  "start_coord", 
                  "end_coord", 
                  "sequence", 
                  "raw_control_coverage", 
                  "raw_treatment_coverage", 
                  "norm_control_coverage", 
                  "norm_treatment_coverage"]

    # keep rows with "Chr" in the chromosome column
    df = df[df.ref.isin(["Chr"+str(i) for i in range(1,6)])]  

    # get rid of rows with non-standard characters
    odds = [s for s in list(set("".join(df.sequence))) if s not in ["A", "T", "C", "G"]]
    if len(odds) > 0:
      mask = df.sequence.str.contains("|".join(odds))  # true if contains weird characters, false if contains only ATCG
      df = df[np.logical_not(mask)]  # keep only rows without weird characters 

    # get rid of rows with raw_control_coverage < x and raw_treatment_coverage < y
    df = df[(df.raw_control_coverage >= args.control_threshold) & (df.raw_treatment_coverage >= args.treatment_threshold)]

    # create target column
    df["target"] = np.log2(df.norm_control_coverage/df.norm_treatment_coverage)

    # bring sequences up to desired length using Xs
    if args.pad_up_to > 0:
      difference = args.pad_up_to - len(df.sequence.tolist()[0])
      assert difference >= 0  # only contains functionality to pad upwards
      df["sequence"] += "X"*difference

    # assign sets
    df["set"] = "train"
    picked = ["Chr2","Chr4"]  # candidates for val and test
    random.Random(1202).shuffle(picked)
    # set val and test by chromosome we want
    df.loc[df.ref == picked[0], "set"] = "val"
    df.loc[df.ref == picked[1], "set"] = "test"

    df = df[["ref", "start_coord", "end_coord", "sequence", "target", "set"]]

    df.to_csv(args.data_output_path, index=False)


main()
