# process annotated arabidopsis genome starr dataset, test informed downsampling method

import argparse
import numpy as np
import pandas as pd
import random


def fetch_args():
    parser = argparse.ArgumentParser(description="Unannotated Arabidopsis Genome STARR Dataset Creation")

    parser.add_argument('-di', '--data_input_path', type=str, default="data/raw/athal_starr.acr.regions.bed", help='Path to annotated starr data')
    parser.add_argument('-ao', '--accessible_data_output_path', type=str, default="data/processed/athaliana_genome_starr_accessible.csv", help='Formatted dataset output path')
    parser.add_argument('-io', '--inaccessible_data_output_path', type=str, default="data/processed/athaliana_genome_starr_inaccessible.csv", help='Formatted dataset output path')

    parser.add_argument('-sa', '--sample', type=int, default=1, help='Sample every n rows')
    parser.add_argument('-pa', '--pad_up_to', type=int, default=0, help='Add X to sequences to reach desired length, 0 if leave alone')

    parser.add_argument('-tr', '--trim', type=bool, default=True, help='Whether to enforce no overlapping sequences between accessible and inaccessible datasets')
    parser.add_argument('-ds', '--downsample', type=bool, default=True, help='Whether to force inaccessible sequences distribution to match accessible')
    parser.add_argument('-ss', '--save_separate', type=bool, default=True, help='Whether accessible/inaccessible datasets are saved separately')

    parser.add_argument('-sc', '--control_threshold', type=int, default=30, help='Raw control coverage threshold')
    parser.add_argument('-st', '--treatment_threshold', type=int, default=5, help='Raw treatment coverage threshold')

    args = parser.parse_args()

    return args


def trim_and_downsample(df, args):
    '''
    Separates dataset into accessible and inaccessible that contain no overlapping sequences
    Optionally downsamples inaccessible rows to exactly match the lengths distribution of accessible rows
    Hard-coded for step size of 17, sequence length 153
    '''

    # indicates where to split based on row removal due to insufficient reads
    split_reads = [0] + [1 if df.start_coord.iloc[i] - df.start_coord.iloc[i-1] > 17 else 0 for i in range(1, df.shape[0])]

    # indicates where to split based on row removal due to accessibility bit changing 0->1 or 1->0
    split_accessible = [0] + [1 if df.ACR_flag.iloc[i] != df.ACR_flag.iloc[i-1] else 0 for i in range(1, df.shape[0])]

    df["split"] = np.logical_or(split_reads, split_accessible).astype(int)  # creates one column where everything before the 1 bit is added to chunk list 

    chunks = split_into_chunks(df)

    # split chunks based on accessible vs inaccessible
    acc_chunks = []
    inacc_chunks = []
    for chunk in chunks:
      if chunk.ACR_flag.iloc[0] == 1:
        acc_chunks.append(chunk)
      elif chunk.ACR_flag.iloc[0] == 0:
        inacc_chunks.append(chunk)

    # get rid of 4 rows on each side to ensure no overlaps
    inacc_chunks_trimmed = trim(inacc_chunks)
    acc_chunks_trimmed = trim(acc_chunks)

    df_inacc = pd.concat(inacc_chunks_trimmed).sort_values(["ref", "start_coord"])
    df_acc = pd.concat(acc_chunks_trimmed).sort_values(["ref", "start_coord"])

    if not args.downsample:
      # not looking to downsample inaccessible chunks
      return df_acc, df_inacc

    # force inacc_chunks_trimmed_lens to match acc_chunks_trimmed_lens
    acc_chunks_trimmed_lens = [chunk.shape[0] for chunk in acc_chunks_trimmed]
    lengths = sorted(acc_chunks_trimmed_lens, reverse=True)
    final_inacc_dataset = []
    for length in lengths:
      # get list of chunks that are longer than the current accessible chunk length we're considering
      long_inacc_chunks = [chunk for chunk in inacc_chunks_trimmed if chunk.shape[0] >= length]  # create subset of inacc_chunks_trimmed where len(all chunks) > length

      # randomly select chunk from list
      chunk = random.choice(long_inacc_chunks)

      # randomly select subchunk from chunk
      subchunks = split_thricely(chunk, length)

      # add subchunk to final dataset
      final_inacc_dataset.append(subchunks[0])

      # remove chunk from inacc_chunks_trimmed based on first index position
      subchunk_indentifier = subchunks[0].index[0]
      inacc_chunks_trimmed = [chunk for chunk in inacc_chunks_trimmed if chunk.index[0] != subchunk_indentifier]

      # add others back in if they exist
      if len(subchunks) > 1:
        inacc_chunks_trimmed += subchunks[1:]

    df_inacc = pd.concat(final_inacc_dataset).sort_values(["ref", "start_coord"])

    return df_acc, df_inacc


def split_into_chunks(df):  
    '''
    Takes in df with attribute "split", return list of dataset chunks
    Split attribute contains a 1 in rows taht directly succeed the boundary 
      where the dataframe is to be split
    '''

    last_1_idx = 0
    chunks = []
    split_list = df.split.tolist()

    # wherever there is a 1 in df.split
    for i in range(len(split_list)): 
      if split_list[i] == 1:
        current_1_idx = i
        chunks.append(df.iloc[last_1_idx:current_1_idx])
        last_1_idx = current_1_idx

    return chunks


def split_thricely(chunk, length):  
    '''
    Given a df chunk, select a subchunk that is length rows long
    Return the chosen chunk, as well as the remaining non-chosen rows above and below
    '''

    chunk_size = chunk.shape[0]

    if chunk_size == length:
      return [chunk]
    else:
      chunks_list = []

      # calculate chunk length - length
      start_idx = random.randint(0, chunk_size-length)

      # create actual subchunk
      chunks_list.append(chunk.iloc[start_idx:start_idx+length])

      # add left subchunk and right subchunk if they exist
      left = chunk.iloc[0:start_idx]
      right = chunk.iloc[start_idx+length:chunk_size]

      if left.shape[0] > 0:  # if there's a left subchunk of length >= 1 to add
        chunks_list.append(left)
      if right.shape[0] > 0:  # if there's a right subchunk of length >= 1 to add
        chunks_list.append(right)

      # print([chunk.shape[0] for chunk in chunks_list])
      return chunks_list


def trim(chunks_list):
  '''
  Take in list of chunks of dataframe
  For each chunk, delete rows on each end such that there are no overlaps with neighbouring chunks
  '''

  trimmed_list = []
  for chunk in chunks_list:
    if chunk.shape[0] > 8:  # that is, there will be something left after trimming
      trimmed_list.append(chunk.iloc[4:-4,:])

  return trimmed_list


def main():
    args = fetch_args()  # get args

    df = pd.read_csv(args.data_input_path, sep="\t", skiprows=lambda x: x % args.sample, header=None)
    df.columns = ["ref",
                  "start_coord",
                  "end_coord",
                  "sequence",
                  "control_raw_value", 
                  "starr_raw_value", 
                  "control_normalized", 
                  "starr_normalized",
                  "ACR_flag",
                  "region"]

    # pre-process
    df = df[df.ref.isin(["Chr"+str(i) for i in range(1,6)])]  # keep rows with "Chr" in the chromosome column

    # get rid of rows with non-standard characters
    odds = [s for s in list(set("".join(df.sequence))) if s not in ["A", "T", "C", "G"]]
    if len(odds) > 0:
      mask = df.sequence.str.contains("|".join(odds))  # true if contains weird characters, false if contains only ATCG
      df = df[np.logical_not(mask)]  # keep only rows without weird characters 

    # get rid of rows with raw_control_coverage < x and raw_treatment_coverage < y
    df = df[(df.control_raw_value >= args.control_threshold) & (df.starr_raw_value >= args.treatment_threshold)]

    # create target column
    df["target"] = np.log2(df.control_normalized/df.starr_normalized)

    # bring sequences up to desired length using Xs
    if args.pad_up_to > 0:
      difference = args.pad_up_to - len(df.sequence.tolist()[0])
      assert difference >= 0  # only contains functionality to pad upwards
      df["sequence"] += "X"*difference

    # define sets
    df["set"] = "train"
    picked = ["Chr2","Chr4"]  # candidates for val and test
    random.Random(1202).shuffle(picked)
    # set val and test by chromosome we want
    df.loc[df.ref == picked[0], "set"] = "val"
    df.loc[df.ref == picked[1], "set"] = "test"

    # restrict based on accessibility
    if not args.trim:
      inaccessible = df[df.ACR_flag == 0]
      accessible = df[df.ACR_flag == 1]
    else:
      accessible, inaccessible = trim_and_downsample(df, args)
    
    accessible = accessible[["ref", "start_coord", "end_coord", "sequence", "ACR_flag", "region", "target", "set"]]
    inaccessible = inaccessible[["ref", "start_coord", "end_coord", "sequence", "ACR_flag", "region", "target", "set"]]

    if not args.save_separate:
      # combine accessible and inaccessible to get final dataset
      df = pd.concat([accessible, inaccessible])
      single_path = args.accessible_data_output_path.replace("_accessible", "")
      df.to_csv(single_path, index=False)

    accessible.to_csv(args.accessible_data_output_path, index=False)
    inaccessible.to_csv(args.inaccessible_data_output_path, index=False)


main()
