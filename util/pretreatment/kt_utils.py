import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle

import csv
import os

###################################################################################
###  This module extract of the train_cat-vs-dog.hdf5 and test_cat-vs-dog.hdf5  ###
###  cat-vs-dog.hdf5 for the train and test.                                    ###
###                                                                       UL    ###
###################################################################################

def load_dataset(typeDataset = None, mode = None, dataset = None, train = None, test = None, params = None):
    try:
        if typeDataset == "h5" or typeDataset == "csv":
            print("Datasets used to train and test the neural network")
            # Validating the dataset path
            if dataset is None:
                if(mode == 'train-test'):
                    if(train is None):
                        raise Exception("Error, the property indicating the path of the train dataset is not defined.")
                    if(test is None):
                        raise Exception("Error, the property that indicates the path of the test dataset is not defined.")
                else:
                    if(mode == 'train-train'):
                        if(train is None):
                            raise Exception("Error, the property indicating the path of the train dataset is not defined.")
                    else:
                        raise Exception("Error, the application does not support the defined property.")
            # Depending on the mode, it extracts the dataset and places it in variables
            if typeDataset == "h5":
                if dataset is None:
                    print("train_dataset: "+train)
                    train_dataset = h5py.File(train, "r")
                    train_set_x_orig = np.array(train_dataset[params["train_set_x"]][:]) # your train set features
                    train_set_y_orig = np.array(train_dataset[params["train_set_y"]][:]) # your train set labels

                    if (mode == 'train-test'): # Checking if the variable is None
                        print("test_dataset: "+test)
                        test_dataset = h5py.File(test, "r")
                        test_set_x_orig = np.array(test_dataset[params["test_set_x"]][:]) # your test set features
                        test_set_y_orig = np.array(test_dataset[params["test_set_y"]][:]) # your test set labels
                    else:
                        print("test_dataset: "+train)
                        test_dataset = h5py.File(train, "r")
                        test_set_x_orig = np.array(test_dataset[params["train_set_x"]][:]) # your test set features
                        test_set_y_orig = np.array(test_dataset[params["train_set_y"]][:]) # your test set labels

                    classes = np.array(test_dataset[params["list_classes"]][:]) # the list of classes
                    
                    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
                    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
                else:
                    print("dataset: "+dataset)
                    tdataset = h5py.File(dataset, "r")
                    classes = tdataset[params["list_classes"]][:]

                    tdataset_x_orig = np.array(tdataset[params["train_set_x"]][:]) # your train set features
                    tdataset_y_orig = np.array(tdataset[params["train_set_y"]][:]) # your train set labels

                    train_set_x_orig, test_set_x_orig, train_set_y_orig, test_set_y_orig = train_test_split(tdataset_x_orig, tdataset_y_orig, test_size=0.33, random_state=0, shuffle=True)
                    
                    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
                    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
            else: 
                if typeDataset == "csv":
                    encoder = LabelEncoder()
                    if dataset is None:
                        print("train_dataset: "+train)
                        dataframe = pandas.read_csv(str(train), header=None)
                        tdataset = dataframe.values
                        train_set_x_orig = tdataset[:,0:int(params["cant_input"])].astype(float)
                        train_set_y_orig = tdataset[:,int(params["cant_input"])]

                        if (mode == 'train-train'):
                            print("test_dataset1: "+train)
                            test_set_x_orig = train_set_x_orig
                            test_set_y_orig = train_set_y_orig
                        else:
                            print("test_dataset: "+test)
                            dataframe = pandas.read_csv(str(test), header=None)
                            tdataset = dataframe.values
                            test_set_x_orig = tdataset[:,0:int(params["cant_input"])].astype(float)
                            test_set_y_orig = tdataset[:,int(params["cant_input"])]

                            #X_sparse_test = coo_matrix(test_set_x_orig)
                            #test_set_x_orig, X_sparse_test, test_set_y_orig = shuffle(test_set_x_orig, X_sparse_test, test_set_y_orig, random_state=0)

                        encoder.fit(train_set_y_orig)

                        encoded_test_Y = encoder.transform(test_set_y_orig)
                        test_set_y_orig = encoded_test_Y

                        encoded_train_Y = encoder.transform(train_set_y_orig)
                        train_set_y_orig = encoded_train_Y
                        

                        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
                        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
                        classes = None
                    else:
                        print("dataset: "+dataset) 
                        dataframe = pandas.read_csv(str(dataset), header=None)
                        tdataset = dataframe.values
                        X = tdataset[:,0:int(params["cant_input"])].astype(float)
                        Y = tdataset[:,int(params["cant_input"])]

                        #X_sparse = coo_matrix(X)
                        #X, X_sparse, Y = shuffle(X, X_sparse, Y, random_state=0)

                        encoder.fit(Y)
                        encoded_Y = encoder.transform(Y)
                        Y = encoded_Y
                        
                        classes = None

                        if(mode == 'train-train'):
                            train_set_x_orig = X
                            train_set_y_orig = Y
                            test_set_x_orig  = X
                            test_set_y_orig  = Y
                        else:
                            train_set_x_orig, test_set_x_orig, train_set_y_orig, test_set_y_orig = train_test_split(X, Y, test_size=0.33, random_state=None, shuffle=True)

                        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
                        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        else:
            raise Exception("Error, the dataset type entered is not supported.")

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    except:
        raise Exception("Error, trying to extract the dataset to train and test the dataset.")
