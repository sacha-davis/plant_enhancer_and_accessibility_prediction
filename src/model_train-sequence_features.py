# train model that uses sequence features to predict real-valued or binary target(s)

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
from keras.layers import Input, Dense, Conv1D, Dropout, Flatten, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam  # https://stackoverflow.com/questions/62707558/importerror-cannot-import-name-adam-from-keras-optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping  # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/

from sklearn.metrics import r2_score
from scipy.stats import spearmanr  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

warnings.filterwarnings('ignore')
logging.disable(1000)

tf.random.set_seed(1202)  # https://www.tensorflow.org/api_docs/python/tf/random/set_seed

# for one-hot-encoding
mapping = {"A": [1.0, 0.0, 0.0, 0.0], "T": [0.0, 0.0, 0.0, 1.0], "C": [0.0, 1.0, 0.0, 0.0], "G": [0.0, 0.0, 1.0, 0.0], "X":[0.0, 0.0, 0.0, 0.0]}  # cross referenced with kipoi data loader


def fetch_args():
    parser = argparse.ArgumentParser(description="Model that uses sequences as input")

    parser.add_argument('-da', '--data_input_path', type=str, default="data/processed/athaliana_genome_starr.csv", help='Path to training data')
    parser.add_argument('-sp', '--save_path', type=str, default="experiments/", help='Path to folder to create save folder in')

    parser.add_argument('-ml', '--model', type=str, default="cnn", help='cnn for convolutional neural network, rnn for recurrent neural network')
    parser.add_argument('-ta', '--task', type=str, default="regression", help='regression when predicting real value, binclass for binary classification')

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.002, help='Learning Rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=512, help='Batch Size')
    parser.add_argument('-ep', '--num_epochs', type=int, default=500, help='Total Number of Epochs')
    parser.add_argument('-pa', '--patience', type=int, default=20, help='Patience for early stopping')

    parser.add_argument('-sh', '--shuffle', type=int, default=0, help='1 to shuffle input sequences, 0 to not')

    # cnn-specific arguments
    parser.add_argument('-mp', '--model_path', type=str, default='models/model.json', help='Path to MPRA-DragoNN model')
    parser.add_argument('-wp', '--weights_path', type=str, default='models/pretrained.hdf5', help='Path to MPRA-DragoNN weights')

    parser.add_argument('-c1', '--conv_1_set', type=int, default=0, help='Treatment for first conv layer. 0 = scratch, 1 = starting point, 2 = freeze')
    parser.add_argument('-c2', '--conv_2_set', type=int, default=0, help='Treatment for second conv layer. 0 = scratch, 1 = starting point, 2 = freeze')
    parser.add_argument('-c3', '--conv_3_set', type=int, default=0, help='Treatment for third conv layer. 0 = scratch, 1 = starting point, 2 = freeze')

    parser.add_argument('-lm', '--linear_mapping', type=int, default=0, help='1 to build on top of model output, 0 to replace')
    parser.add_argument('-lc', '--last_conv_layer', type=int, default=1, help='1 to keep last conv layer, 0 to delete it')

    # rnn-specific arguments
    parser.add_argument('-ls', '--layer_size', type=int, default=128, help='LSTM layer size')

    args = parser.parse_args()

    return args


def train_test_val(args, df, target_cols):
    '''
    Define X and y for train, val, and set sets
    X is made up of one-hot-encoded sequences
    y is the matrix of target columns
    '''

    if args.shuffle == 1:  # shuffles NTs within each sequence
      df.loc[:,"sequence"] = [''.join(random.sample(s, len(s))) for s in df["sequence"]]

    train_df = df[df.set == "train"]
    X_train = np.array([get_ohe(sqnc) for sqnc in train_df["sequence"]])
    y_train = np.array(train_df[target_cols])

    val_df = df[df.set == "val"]
    X_val = np.array([get_ohe(sqnc) for sqnc in val_df["sequence"]])
    y_val = np.array(val_df[target_cols])

    test_df = df[df.set == "test"]
    X_test = np.array([get_ohe(sqnc) for sqnc in test_df["sequence"]])
    y_test = np.array(test_df[target_cols])

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_ohe(sequence): 
    '''
    One-hot-encodes sequence to format model can use: (len(sequence), 4)
    '''
    return np.array([mapping[nt] for nt in sequence])


def get_cnn_model(args, in_dim, out_dim):  # initializes model architecture
    '''
    Initialize cnn model
    Structure and activations are as per MPRA-DragoNN, which is inspired by the basset architecture
    These parameters must be the consistent to use weights in transfer learning task
    '''

    mdl = Sequential()

    conv1_train = args.conv_1_set != 2  # True if conv layer should be trained
    mdl.add(Conv1D(120, 5, activation='relu', input_shape=(in_dim, 4), name="1DConv_1", trainable=conv1_train))
    mdl.add(BatchNormalization(name="batchNorm1", trainable=conv1_train))
    mdl.add(Dropout(0.1, name="drop1"))

    conv2_train = args.conv_2_set != 2  # True if conv layer should be trained
    mdl.add(Conv1D(120, 5, activation='relu', name="1DConv_2", trainable=conv2_train))
    mdl.add(BatchNormalization(name="batchNorm2", trainable=conv2_train))
    mdl.add(Dropout(0.1, name="drop2"))

    if args.last_conv_layer == 1:  # if we are not removing last conv layer for simplicity
      conv3_train = args.conv_3_set != 2  # True if conv layer should be trained
      mdl.add(Conv1D(120, 5, activation='relu', name="1DConv_3", trainable=conv3_train))
      mdl.add(BatchNormalization(name="batchNorm3", trainable=conv3_train))
      mdl.add(Dropout(0.1, name="drop3"))

    mdl.add(Flatten(name="flat"))

    if args.linear_mapping == 1:  
        # if true and other conv_sets are frozen, then we effectively only train 12 weights
        mdl.add(Dense(12, activation='linear', name="dense1", trainable=False))

    # output layer
    output_layer_activation = "linear" if args.task == "regression" else "sigmoid"
    mdl.add(Dense(out_dim, activation=output_layer_activation, name="dense2"))

    return mdl


def set_cnn_weights(args, pretrained_mdl, mdl):  # sets appropriate model weights from pretrained
    '''
    Map weights from specific layers of pretrained cnn model to current cnn model
    Allows us more granular control over which layers get preset
    '''

    layers_to_set = []  # contains indices of layers to set
    if args.conv_1_set != 0: layers_to_set += [0, 1, 2]
    if args.conv_2_set != 0: layers_to_set += [3, 4, 5]
    if args.conv_3_set != 0: layers_to_set += [6, 7, 8]

    for i in layers_to_set:
        pretrained_layer_weights = pretrained_mdl.layers[i].get_weights()  # get pre-trained layer weights
        mdl.layers[i].set_weights(pretrained_layer_weights)  # set layer weights

    return mdl


def get_rnn_model(args, in_dim, out_dim):
    '''
    Initialize rnn model
    '''
    mdl = keras.Sequential()

    # only tested this with one LSTM layer
    mdl.add(LSTM(args.layer_size))

    # output layer
    output_layer_activation = "linear" if args.task == "regression" else "sigmoid"
    mdl.add(Dense(out_dim, activation=output_layer_activation))

    return mdl


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

    df = pd.read_csv(args.data_input_path)
    target_cols = [col for col in df.columns if "target" in col]  # contains names of columns to predict

    print("targets:", target_cols)

    X_train, y_train, X_val, y_val, X_test, y_test = train_test_val(args, df, target_cols)
    print("got data")

    if args.model == "cnn": 
        # create path to folder with results 
        dir_path = (args.save_path
                    +args.model
                    +"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    +"_out"+str(len(target_cols))  # number of targets being predicted
                    +"_lr"+str(args.learning_rate)
                    +"_bs"+str(args.batch_size)
                    +"_sh"+str(args.shuffle)
                    +"_"+str(args.conv_1_set)
                    +str(args.conv_2_set)
                    +str(args.conv_3_set)
                    +str(args.linear_mapping))
        # define model
        pretrained_model = load_pretrained_model(args.model_path, args.weights_path)  # load in MPRA-DragoNN model arch and weights
        model = get_cnn_model(args, X_train.shape[1], len(target_cols))  

        model = set_cnn_weights(args, pretrained_model, model)  # set appropriate weights from pretrained_model to model
    elif args.model == "rnn":
        # create path to folder with results 
        dir_path = (args.save_path
                    +args.model
                    +"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    +"_out"+str(len(target_cols))  # number of targets being predicted
                    +"_lr"+str(args.learning_rate)
                    +"_bs"+str(args.batch_size)
                    +"_sh"+str(args.shuffle)
                    +"_ls"+str(args.layer_size))
        # define model
        model = get_rnn_model(args, X_train.shape[1], len(target_cols)) 
    else: print("You shouldn't be here, check model definition") 

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
    saved_model = load_pretrained_model(dir_path+"/model_architecture.json", dir_path+"/best_weights.h5")

    # evaluate model
    if args.task == "regression": real_value_evaluation(dir_path, target_cols, saved_model, X_train, X_val, X_test, y_train, y_val, y_test)
    elif args.task == "binclass": binary_evaluation(dir_path, target_cols, saved_model, X_train, X_val, X_test, y_train, y_val, y_test)
    else: print("You shouldn't be here, check task definition")

    # write all args to text file for reproducibility 
    json.dump(vars(args), open(dir_path+"/settings.txt", "w"))  # https://www.kite.com/python/answers/how-to-save-a-dictionary-to-a-file-in-python


main()
