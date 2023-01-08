# train feed-forward neural network to predict real-valued or binary target(s)

import argparse
import keras
import warnings, logging
import numpy as np
import pandas as pd
import datetime, time, os
import json
import random
import tensorflow as tf

from model_helpers import *
from keras.models import Sequential, model_from_json
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam  # https://stackoverflow.com/questions/62707558/importerror-cannot-import-name-adam-from-keras-optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping  # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
logging.disable(1000)

tf.random.set_seed(1202)  # https://www.tensorflow.org/api_docs/python/tf/random/set_seed

nts = ["A", "T", "C", "G"]  # list of single nucleotides


def fetch_args():
    parser = argparse.ArgumentParser(description="Model that uses frequency features as input")

    parser.add_argument('-da', '--data_input_path', type=str, default="data/processed/athaliana_genome_starr.csv", help='Path to training data')
    parser.add_argument('-sp', '--save_path', type=str, default="experiments/", help='Path to folder to create save folder in')

    parser.add_argument('-ml', '--model', type=str, default="ffnn", help='legacy, do not change value')
    parser.add_argument('-ta', '--task', type=str, default="regression", help='regression when predicting real value, binclass for binary classification')

    parser.add_argument('-mo', '--include_mononuc_freq', type=int, default=1, help='Use single nucleotide frequencies')
    parser.add_argument('-di', '--include_dinuc_freq', type=int, default=0, help='Use dinucleotide frequencies')
    parser.add_argument('-tr', '--include_trinuc_freq', type=int, default=0, help='Use trinucleotide frequencies')

    # ffnn-specific arguments
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.002, help='Learning Rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch Size')
    parser.add_argument('-ep', '--num_epochs', type=int, default=100, help='Total Number of Epochs')
    parser.add_argument('-pa', '--patience', type=int, default=20, help='Patience for early stopping')

    parser.add_argument('-l1', '--layer_1_size', type=int, default=12, help='Num nodes in hidden layer 1')
    parser.add_argument('-a1', '--layer_1_activation', type=str, default="relu", help='Activation for hidden layer 1')
    parser.add_argument('-l2', '--layer_2_size', type=int, default=0, help='Num nodes in hidden layer 2')
    parser.add_argument('-a2', '--layer_2_activation', type=str, default="relu", help='Activation for hidden layer 2')
    parser.add_argument('-l3', '--layer_3_size', type=int, default=0, help='Num nodes in hidden layer 3')
    parser.add_argument('-a3', '--layer_3_activation', type=str, default="relu", help='Activation for hidden layer 3')
    # (output size and activation are automatically inferred from data and task argument)

    args = parser.parse_args()

    return args


def get_ffnn_model(args, in_dim, out_dim):
    '''
    Initializes feed-forward model architecture
    '''

    mdl = Sequential()

    # this is the only layer that is enforced. to test linear regression only, set layer_1_size to 1 and layer_1_activation to "linear"
    mdl.add(Dense(args.layer_1_size, input_dim=in_dim, activation=args.layer_1_activation))

    if args.layer_2_size > 0:       mdl.add(Dense(args.layer_2_size, activation=args.layer_2_activation))
    if args.layer_3_size > 0:       mdl.add(Dense(args.layer_3_size, activation=args.layer_3_activation))

    # add appropriate output layer
    output_layer_activation = "linear" if args.task == "regression" else "sigmoid"
    mdl.add(Dense(out_dim, activation=output_layer_activation))

    return mdl


def train_test_val(args, df, target_cols):
    '''
    Define X and y for train, val, and set sets
    X is made up of some combination of mononucleotide, dinucleotide, and/or trinucleotide frequencies
    y is the matrix of target columns
    '''

    include = []  # captures all sequences we are including as input features
    if args.include_mononuc_freq == 1:  include += nts
    if args.include_dinuc_freq == 1:    include += [nt1+nt2 for nt1 in nts for nt2 in nts]
    if args.include_trinuc_freq == 1:   include += [nt1+nt2+nt3 for nt1 in nts for nt2 in nts for nt3 in nts]

    for item in include:  # create new columns with the counts of sequences in "include"
      df[item] = df.sequence.str.count(item)

    train_df = df[df.set == "train"]
    X_train = np.array(train_df[include])
    y_train = np.array(train_df[target_cols])

    val_df = df[df.set == "val"]
    X_val = np.array(val_df[include])
    y_val = np.array(val_df[target_cols])

    test_df = df[df.set == "test"]
    X_test = np.array(test_df[include])
    y_test = np.array(test_df[target_cols])

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    args = fetch_args()  # get args

    df = pd.read_csv(args.data_input_path)
    target_cols = [col for col in df.columns if "target" in col]  # contains names of columns to predict

    print("targets:", target_cols)

    X_train, y_train, X_val, y_val, X_test, y_test = train_test_val(args, df, target_cols)
    print("got data")

    # create path to folder with results 
    dir_path = (args.save_path
                +args.model
                +"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                +"_nuc"
                +str(args.include_mononuc_freq)
                +str(args.include_dinuc_freq)
                +str(args.include_trinuc_freq)
                +"_lay"+str(args.layer_1_size)
                +"-"+str(args.layer_2_size)
                +"-"+str(args.layer_3_size)
                +"-"+str(len(target_cols))
                +"_lr"+str(args.learning_rate)
                +"_bs"+str(args.batch_size))

    # define model
    model = get_ffnn_model(args, X_train.shape[1], len(target_cols))  

    # compile model
    model.compile(optimizer=Adam(lr=args.learning_rate),  # CHANGE IF WE WANT TO CHANGE OPTIM
                  loss='mean_squared_error' if args.task == "regression" else "binary_crossentropy",
                  metrics=[Spearman if args.task == "regression" else tf.keras.metrics.AUC()])

    # init callbacks
    logdir = os.path.join(dir_path, "logs")
    tensorboard_callback = keras.callbacks.TensorBoard(logdir, histogram_freq=1)  # https://stackoverflow.com/questions/59894720/keras-and-tensorboard-attributeerror-sequential-object-has-no-attribute-g
    es_callback = EarlyStopping(monitor='val_loss', verbose=1, patience=args.patience)
    mc_callback = ModelCheckpoint(dir_path+'/best_weights.h5', monitor='val_loss', save_best_only=True)

    # train model
    history = model.fit(X_train, y_train,
                        epochs=args.num_epochs,
                        batch_size=args.batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[tensorboard_callback, es_callback, mc_callback])

    # save training history
    hist_df = pd.DataFrame(history.history) 
    hist_df.to_csv(dir_path+'/training_history.csv')

    # save model architecture
    model_json = model.to_json()
    with open(dir_path+"/model_architecture.json", "w") as json_file:
        json_file.write(model_json)

    # load json and create model
    with open(dir_path+"/model_architecture.json", "r") as json_file:
        loaded_model_json = json_file.read()
    saved_model = model_from_json(loaded_model_json)
    # load weights into model
    saved_model.load_weights(dir_path+"/best_weights.h5")

    # evaluate model
    if args.task == "regression": real_value_evaluation(dir_path, target_cols, saved_model, X_train, X_val, X_test, y_train, y_val, y_test)
    elif args.task == "binclass": binary_evaluation(dir_path, target_cols, saved_model, X_train, X_val, X_test, y_train, y_val, y_test)
    else: print("You shouldn't be here, check task definition")

    # write all args to text file for reproducibility 
    json.dump(vars(args), open(dir_path+"/settings.txt", "w"))  # https://www.kite.com/python/answers/how-to-save-a-dictionary-to-a-file-in-python


main()
