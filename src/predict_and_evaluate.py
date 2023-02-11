# generate predictions (and epotentially evaluations) given a model and a dataset

import argparse
import keras
import warnings, logging
import numpy as np
import pandas as pd
import datetime, time, os
import json
import random
import tensorflow as tf
import ast
import os

from model_helpers import *
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Input, Dense, Conv1D, Dropout, Flatten, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam  # https://stackoverflow.com/questions/62707558/importerror-cannot-import-name-adam-from-keras-optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping  # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

from sklearn.metrics import r2_score
from scipy.stats import spearmanr  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

warnings.filterwarnings('ignore')
logging.disable(1000)

tf.random.set_seed(1202)  # https://www.tensorflow.org/api_docs/python/tf/random/set_seed

nts = ["A", "T", "C", "G"] 
mapping = {"A": [1.0, 0.0, 0.0, 0.0], "T": [0.0, 0.0, 0.0, 1.0], "C": [0.0, 1.0, 0.0, 0.0], "G": [0.0, 0.0, 1.0, 0.0], "X":[0.0, 0.0, 0.0, 0.0]}  # cross referenced with kipoi data loader



def fetch_args():
    parser = argparse.ArgumentParser(description="Evaluate frequency-based or sequence-based model on specified dataset")

    parser.add_argument('-da', '--data_input_path', type=str, default="data/processed/ntabacum_35s_starr.csv", help='Path to training data')
    parser.add_argument('-ep', '--experiment_path', type=str, default="models/cnn_20230211-200556_out1_lr0.002_bs512_sh0_0000", help='Path to folder with model and settings')
    parser.add_argument('-sp', '--save_path', type=str, default="evaluation/", help='Path to folder to create save folder in')

    args = parser.parse_args()

    return args


def prepare_dataset_freq(experiment_args, df, target_cols):
    '''
    Prepare X and y to test model on that uses frequency features as input
    '''
    include = []  # captures all sequences we are including as input features

    if experiment_args["include_mononuc_freq"] == 1:  include += nts
    if experiment_args["include_dinuc_freq"] == 1:    include += [nt1+nt2 for nt1 in nts for nt2 in nts]
    if experiment_args["include_trinuc_freq"] == 1:   include += [nt1+nt2+nt3 for nt1 in nts for nt2 in nts for nt3 in nts]

    for item in include:  # create new columns with the counts of sequences in "include"
      df[item] = df.sequence.str.count(item)

    X = np.array(df[include])

    # if target_cols is empty, we just want to generate predictions and not evaluate
    if len(target_cols) > 0:  y = np.array(df[target_cols])
    else:  y = None

    return X, y, include


def prepare_dataset_seq(df, target_cols):
    '''
    Prepare X and y to test model on that uses sequence as input
    '''
    X = np.array([get_ohe(sqnc) for sqnc in df["sequence"]])

    # if target_cols is empty, we just want to generate predictions and not evaluate
    if len(target_cols) > 0:  y = np.array(df[target_cols])
    else:  y = None

    return X, y


def get_ohe(sequence): 
    '''
    One-hot-encodes sequence to format model can use: (len(sequence), 4)
    '''
    return np.array([mapping[nt] for nt in sequence])


def load_pretrained_model(model_path, weights_path):
    '''
    Loads model architecture from json file, sets weights
    '''
    with open(model_path, "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into model
    loaded_model.load_weights(weights_path)

    return loaded_model


def main():
    args = fetch_args()  # get args

    if not os.path.exists(args.save_path):
      os.mkdir(args.save_path)

    # load in args file from settings.txt
    with open(args.experiment_path+'/settings.txt') as f:  # load in settings
      experiment_args = ast.literal_eval(f.read())

    # load data
    df = pd.read_csv(args.data_input_path)
    target_cols = [col for col in df.columns if "target" in col]  # empty if there isn't any ground truth

    print("targets:", target_cols)

    # load pretrained model 
    model = load_pretrained_model(args.experiment_path+'/model_architecture.json', args.experiment_path+'/best_weights.h5')

    # generate OOD X, y
    if experiment_args["model"] in ["ffnn"]:
      X, y, features = prepare_dataset_freq(experiment_args, df, target_cols)
    elif experiment_args["model"] in ["cnn", "rnn"]:
      X, y, features = prepare_dataset_seq(experiment_args, df, target_cols)
    else:
      print("You shouldn't be here, check experiment model definition")

    # clear df of columns we don't want to save
    df = df[[col for col in df.columns if col not in features]]

    # generate y_pred based on X
    y_pred = model.predict(X)

    # evaluate model & save results
    if len(target_cols) > 0:
      for i, name in enumerate(target_cols):
        if experiment_args["task"] == "regression":  # use real-value evaluation metrics
          df[name+"_pred"] = y_pred[:,i]
          # evaluate 
          with open(args.save_path+"results_"+name+".csv", "w") as f:
            f.write("metric,score\n")
            f.write("r2,"+str(r2_score(y[:,i], y_pred[:,i]))+"\n")
            f.write("spearman,"+str(spearmanr(y[:,i], y_pred[:,i])[0]))

        elif experiment_args["task"] == "binclass":  # use binary class evaluation metrics
          y_prob = y_pred[:,i]
          df[name+"_prob"] = y_prob

          y_class = (y_prob > 0.5).astype("int32")
          df[name+"_pred"] = y_class

          # evaluate
          with open(args.save_path+"results_"+name+".csv", "w") as f:
            results = get_results(y, y_pred)
            f.write("metric,score\n")
            f.write("accuracy,"+str(accuracy_score(y[:,i], y_class))+"\n")
            f.write("AUC,"+str(roc_auc_score(y[:,i], y_prob))+"\n")
            f.write("precision,"+str(results["precision"])+"\n")
            f.write("recall-sensitivity,"+str(results["recall-sensitivity"])+"\n")
            f.write("specificity,"+str(results["specificity"])+"\n")
            f.write("TN,"+str(results["TN"])+"\n")
            f.write("FN,"+str(results["FN"])+"\n")
            f.write("TP,"+str(results["TP"])+"\n")
            f.write("FP,"+str(results["FP"]))

        else: print("You shouldn't be here, check task definition")

    else:  # when we don't have a ground truth to compare to but we want to save the predictions
      for i in range(y_pred.shape[1]):  # for each column in output array
        if experiment_args["task"] == "regression":
          df["target_pred_"+str(i)] = y_pred[:,i]
        elif experiment_args["task"] == "binclass":
          y_prob = y_pred[:,i]
          df["target_prob_"+str(i)] = y_prob
          y_class = (y_prob > 0.5).astype("int32")
          df["target_pred_"+str(i)] = y_class


    # save dataset with new columns
    df_save_path = args.save_path + args.data_input_path.split("/")[-1].replace(".csv", "_with_pred.csv")
    df.to_csv(df_save_path, index=False)

    # write all args to text file for reproducibility 
    json.dump(vars(args), open(args.save_path+"evaluation_settings.txt", "w"))  # https://www.kite.com/python/answers/how-to-save-a-dictionary-to-a-file-in-python


main()
