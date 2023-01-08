# process drosophila starr dataset 

import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats


def fetch_args():
    parser = argparse.ArgumentParser(description="Drosophila STARR Dataset Creation")

    parser.add_argument('-na', '--train_activity', type=str, default="data/raw/Sequences_activity_Train.txt", help='Path to train activity file')
    parser.add_argument('-la', '--val_activity', type=str, default="data/raw/Sequences_activity_Val.txt", help='Path to val activity file')
    parser.add_argument('-ta', '--test_activity', type=str, default="data/raw/Sequences_activity_Test.txt", help='Path to test activity file')

    parser.add_argument('-ns', '--train_sequences', type=str, default="data/raw/Sequences_Train.fa", help='Path to train sequences file')
    parser.add_argument('-ls', '--val_sequences', type=str, default="data/raw/Sequences_Val.fa", help='Path to val sequences file')
    parser.add_argument('-ts', '--test_sequences', type=str, default="data/raw/Sequences_Test.fa", help='Path to test sequences file')

    parser.add_argument('-do', '--data_output_path', type=str, default="data/processed/dmelanogaster.csv", help='Formatted dataset output path')

    parser.add_argument('-tg', '--target', type=str, default="hk", help='hk to keep housekeeping, dev to keep dev, multi to keep both')

    parser.add_argument('-pa', '--pad_up_to', type=int, default=0, help='Add X to sequences to reach desired length, 0 if leave alone')

    args = parser.parse_args()

    return args


def main():
    args = fetch_args()  # get args

    # read in activity data
    activity_train_df = pd.read_csv('data/raw/Sequences_activity_Train.txt', sep="\t")[["Dev_log2_enrichment_quantile_normalized", "Hk_log2_enrichment_quantile_normalized"]]
    activity_test_df = pd.read_csv('data/raw/Sequences_activity_Test.txt', sep="\t")[["Dev_log2_enrichment_quantile_normalized", "Hk_log2_enrichment_quantile_normalized"]]
    activity_val_df = pd.read_csv('data/raw/Sequences_activity_Val.txt', sep="\t")[["Dev_log2_enrichment_quantile_normalized", "Hk_log2_enrichment_quantile_normalized"]]

    # create set attribute
    activity_train_df["set"] = "train"
    activity_test_df["set"] = "test"
    activity_val_df["set"] = "val"

    # read in sequennces
    with open('data/raw/Sequences_Train.fa') as f:
      s = f.read().split("\n")
      sequence_train_list = [s[i] for i in range(len(s)) if i%2==1]
      activity_train_df["sequence"] = sequence_train_list

    with open('data/raw/Sequences_Val.fa') as f:
      s = f.read().split("\n")
      sequence_val_list = [s[i] for i in range(len(s)) if i%2==1]
      activity_val_df["sequence"] = sequence_val_list

    with open('data/raw/Sequences_Test.fa') as f:
      s = f.read().split("\n")
      sequence_test_list = [s[i] for i in range(len(s)) if i%2==1]
      activity_test_df["sequence"] = sequence_test_list

    # create single dataset
    df = pd.concat([activity_train_df, activity_test_df, activity_val_df])
    df = df.rename(columns={"Dev_log2_enrichment_quantile_normalized":"dev_target", "Hk_log2_enrichment_quantile_normalized":"hk_target"})

    # get rid of weird characters
    genome = "".join(df.sequence.tolist())
    mask = [True if "N" in s else False for s in df.sequence]
    df = df.iloc[np.logical_not(mask),:]

    # make sure target is good
    if args.target == "hk":
      df.drop(columns="dev_target")
    elif args.target == "dev":
      df.drop(columns="hk_target")
      # anything else is assumed to be intended as multi-target

    df.to_csv(args.data_output_path, index=False)


main()
