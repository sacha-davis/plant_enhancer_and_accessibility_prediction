# process tobacco 35s starr dataset 

import argparse
import math
import pandas as pd
import numpy as np

# define the 35S sequence that was augmented and tested
thirty_five_s = "AGATCTCTCTGCCGACAGTGGTCCCAAAGATGGACCCCCACCCACGAGGAGCATCGTGGAAAAAGAAGACGTTCCAACCACGTCTTCAAAGCAAGTGGATTGATGTGACATCTCCACTGACGTAAGGGATGACGCACAATCCCACTATCCTTC"


def fetch_args():
    parser = argparse.ArgumentParser(description="Dataset Creation")

    parser.add_argument('-di', '--data_input_path', type=str, default="data/raw/tobacco.csv", help='Path to 35S STARR data')
    parser.add_argument('-do', '--data_output_path', type=str, default="data/processed/ntabacum_35s_starr.csv", help='Formatted dataset output path')

    parser.add_argument('-pa', '--pad_up_to', type=int, default=0, help='Add X to sequences to reach desired length, 0 if leave alone')

    args = parser.parse_args()

    return args


def insertion(base_enhancer, position, nucleotide):
    '''
    Make singular insertion change to sequence
    '''

    base_enhancer = [i for i in base_enhancer]
    base_enhancer.insert(math.floor(position), nucleotide)
    return "".join(base_enhancer)[:-1]  # [:-1] to reach len 153


def deletion(base_enhancer, position):
    '''
    Make singular deletion change to sequence
    '''

    base_enhancer = [i for i in base_enhancer]
    base_enhancer.pop(int(position)-1)
    return "".join(base_enhancer+["X"]) # +["X"] to reach len 153


def substitution(base_enhancer, position, nucleotide):
    '''
    Make singular substitution change to sequence
    '''

    base_enhancer = [i for i in base_enhancer]
    base_enhancer[int(position)-1] = nucleotide
    return "".join(base_enhancer)


def process_change(base_enhancer, df):
    '''
    Run through list of all sequences and generate changed sequences
    '''

    sequences_list = []
    for i in range(df.shape[0]):  # for every row that needs a sequence
      position = df.iloc[i, :]["pos"]
      nucleotide = df.iloc[i, :]["var.nt"]
      if df.iloc[i, :]["type"] == "insertion":
        sequences_list.append(insertion(base_enhancer, position, nucleotide))
      elif df.iloc[i, :]["type"] == "deletion":
        sequences_list.append(deletion(base_enhancer, position))
      else:
        sequences_list.append(substitution(base_enhancer, position, nucleotide))

    df["sequence"] = sequences_list  # add sequence attribute to df

    return df


def main():
    args = fetch_args()  # get args

    df = pd.read_csv(args.data_input_path, header=5)

    df = df[np.logical_not(df.enrichment.isnull())]  # get rid of null enrichment rows

    df = process_change(thirty_five_s, df)  # generate changed sequences

    # TODO: create set attribute? May not be necessary as we haven't needed to train on this data

    # bring sequences up to desired length using Xs
    if args.pad_up_to > 0:
      difference = args.pad_up_to - len(df.sequence.tolist()[0])
      assert difference >= 0  # only contains functionality to pad upwards
      df["sequence"] += "X"*difference

    df = df.rename(columns={"enrichment":"target"})
    df = df[["sequence", "target"]]

    df.to_csv(args.data_output_path, index=False)


main()
