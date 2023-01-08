# process canola genome ATAC dataset w/ binary targets

import argparse
import numpy as np
import pandas as pd
import random


def fetch_args():
    parser = argparse.ArgumentParser(description="Canola Genome ATAC Dataset Creation")

    parser.add_argument('-di', '--data_input_path', type=str, default="data/raw/atac.expression.binary.reformatted.tsv", help='Path to ATAC data')
    parser.add_argument('-do', '--data_output_path', type=str, default="data/processed/bnapus_genome_atac_binary.csv", help='Formatted dataset output path')

    parser.add_argument('-sa', '--sample', type=int, default=1, help='Sample every n rows')

    parser.add_argument('-ta', '--target', type=str, default="multi", help='melted to stack target columns, multi for multi-task prediction, <tissue name picked from samples> for single tissue prediction')

    args = parser.parse_args()

    return args


def main():
    args = fetch_args()  # get args

    df = pd.read_csv(args.data_input_path, sep="\t", skiprows=lambda x: x % args.sample, header=None)  # read in every nth row

    # define column names
    samples =  ["bud-green",
                "bud-yellow",
                "peduncle-down-15cm",
                "seed-21d",
                "silique-1week",
                "silique-2week",
                "silique-3week",
                "silique-4week",
                "stem-down-15cm"]

    column_names = ["ref", 
                    "start_coord", 
                    "end_coord",
                    "sequence"]
                   
    df.columns = column_names + [sample+"_target" for sample in samples]

    # get rid of weirdo rows
    df = df[np.logical_not(df.ref.isin(["chr_contigs", "napus_chloroplast", "napus_mitochondrion"]))]

    odds = [s for s in list(set("".join(df.sequence))) if s not in ["A", "T", "C", "G"]]
    if len(odds) > 0:
      mask = df.sequence.str.contains("|".join(odds))  # true if contains weird characters, false if contains only ATCG
      df = df[np.logical_not(mask)]  # keep only rows without weird characters

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
      df.columns = [column.replace("_target", "") for column in df.columns]  # clean up _norm column names
      df = df.melt(id_vars=["ref", "start_coord", "end_coord", "sequence", "set"],  # https://stackoverflow.com/questions/28654047/convert-columns-into-rows-with-pandas
                   var_name="tissue", 
                   value_name="target")
      df = df[["ref", "start_coord", "end_coord", "sequence", "tissue", "target", "set"]]
    elif args.target in samples:  # normalized reads from one tissue
      df = df.rename(columns={args.target+"_target":"target"})
      df = df[["ref", "start_coord", "end_coord", "sequence", "target", "set"]]
    else:
      print("You shouldn't be here, check target definition")

    df.to_csv(args.data_output_path, index=False)


main()
