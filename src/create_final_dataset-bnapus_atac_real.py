# process canola genome ATAC dataset w/ real-valued targets

import argparse
import numpy as np
import pandas as pd
import random


def fetch_args():
    parser = argparse.ArgumentParser(description="Canola Genome ATAC Dataset Creation")

    parser.add_argument('-di', '--data_input_path', type=str, default="data/raw/atac.expression_with_seq.tsv", help='Path to ATAC data')
    parser.add_argument('-do', '--data_output_path', type=str, default="data/processed/bnapus_genome_atac_real.csv", help='Formatted dataset output path')

    parser.add_argument('-sa', '--sample', type=int, default=1, help='Sample every n rows')

    parser.add_argument('-ta', '--target', type=str, default="averaged", help='averaged to average target columns, melted to stack target columns, multi for multi-task prediction, <tissue picked from samples list> for single tissue prediction')

    parser.add_argument('-ya', '--accessible', type=str, default="all", help='all to include 2% most accessible sequences, none to include none')
    parser.add_argument('-na', '--inaccessible', type=str, default="all", help='all to include 98% least accessible sequences, downsampled to include 2% least accessible sequences, none to include none')

    args = parser.parse_args()

    return args


def main():
    args = fetch_args()  # get args

    df = pd.read_csv(args.data_input_path, sep="\t", skiprows=lambda x: x % args.sample, header=None)  # read in every nth row

    # define column names
    samples = ["bud-green_rep1",
               "bud-green_rep2",
               "bud-yellow_rep1",
               "bud-yellow_rep2",
               "peduncle-down-15cm_rep1",
               "peduncle-down-15cm_rep2",
               "seed-21d_rep1",
               "seed-21d_rep2",
               "silique-1week_rep1",
               "silique-1week_rep2",
               "silique-2week_rep1",
               "silique-2week_rep2",
               "silique-3week_rep1",
               "silique-3week_rep2",
               "silique-4week_rep1",
               "silique-4week_rep2",
               "stem-down-15cm_rep1",
               "stem-down-15cm_rep2"]

    column_names = ["ref", 
                    "start_coord", 
                    "end_coord",
                    "sequence"]
                    
    for item in samples:
      column_names.append(item+"_raw")
      column_names.append(item+"_norm_target")

    df.columns = column_names

    # get rid of weirdo rows
    df = df[np.logical_not(df.ref.isin(["chr_contigs", "napus_chloroplast", "napus_mitochondrion"]))]

    odds = [s for s in list(set("".join(df.sequence))) if s not in ["A", "T", "C", "G"]]
    if len(odds) > 0:
      mask = df.sequence.str.contains("|".join(odds))  # true if contains weird characters, false if contains only ATCG
      df = df[np.logical_not(mask)]  # keep only rows without weird characters

    # create summary columns
    df["norm_read_avg"] = df.iloc[:,[True if "_norm" in s else False for s in df.columns]].mean(axis=1)

    # grab "accessible" sequences we want
    if args.accessible == "all":  # top 2% sequences exclusing top 0.1% outliers
      first_index = int(df.shape[0]*0.979)  # first index we want, hits 97.9%
      final_index = int(df.shape[0]*0.999)  # final index we want, hits 99.9%
      accessible = df.iloc[first_index:final_index,:]
    else: 
      accessible = df.iloc[:0,:]  # empty dataframe

    # grab "inaccessible" sequences we want
    if args.inaccessible == "all":  # bottom 98% sequences
      first_index = 0  # first index we want
      final_index = int(df.shape[0]*0.979)  # final index we want, hits 97.9%
      inaccessible = df.iloc[first_index:final_index,:]
    elif args.inaccessible == "downsampled": # bottom 2% sequences (NOTE: not smartly downsampled)
      first_index = 0  # first index we want
      final_index = int(df.shape[0]*0.02)  # final index we want, hits 2%
      inaccessible = df.iloc[first_index:final_index,:]    
    else: 
      inaccessible = df.iloc[:0,:]  # empty dataframe

    # combine accessible and inaccessible to get final dataset
    df = pd.concat([accessible, inaccessible])
    df = df.sort_values(["ref", "start_coord"])  # makes so we can grab accessible/inaccessible rows by index

    # define sets
    picked1 = ["N"+str(i) for i in range(1,11)]
    picked2 = ["N"+str(i) for i in range(11,20)]
    random.Random(1202).shuffle(picked1) 
    random.Random(1202).shuffle(picked2)

    df["set"] = "train"

    df.loc[df.ref == picked1[0], "set"] = "val"
    df.loc[df.ref == picked2[0], "set"] = "val"

    df.loc[df.ref == picked1[1], "set"] = "test"
    df.loc[df.ref == picked2[1], "set"] = "test"

    # define target(s)
    if args.target == "multi":  # define as multi-task problem
      targets = [column for column in df.columns if "target" in column]
      df = df[["ref", "start_coord", "end_coord", "sequence", "set"]+targets]
    elif args.target == "melted":  # stack all tissues to make really long dataset
      # drop all _raw columns
      df = df[[column for column in df.columns if "_raw" not in column]]  # get rid of _raw columns
      df.columns = [column.replace("_norm_target", "") for column in df.columns]  # clean up _norm column names
      df = df.melt(id_vars=["ref", "start_coord", "end_coord", "sequence", "set", "norm_read_avg"],  # https://stackoverflow.com/questions/28654047/convert-columns-into-rows-with-pandas
                   var_name="tissue", 
                   value_name="target")
      df = df[["ref", "start_coord", "end_coord", "sequence", "tissue", "target", "set"]]
    elif args.target == "averaged":  # averaged normalized reads
      df = df.rename(columns={"norm_read_avg":"target"})
      df = df[["ref", "start_coord", "end_coord", "sequence", "target", "set"]]
    elif args.target in samples:  # normalized reads from one tissue
      df = df.rename(columns={args.target+"_norm_target":"target"})
      df = df[["ref", "start_coord", "end_coord", "sequence", "target", "set"]]

    df.to_csv(args.data_output_path, index=False)


main()
