import pandas as pd
from random import randint
import numpy as np
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing


# Simulated Datasets¶

def load_synthetic_path(data_size, rho):
    """
    TODO: add description
    :param data_size:
    :param rho:
    :return:
    """
    dir_path = "../data/Synthetic/" + data_size + "/rho" + str(rho) + "/"
    brt_path = dir_path
    coeff_path = dir_path
    print("data path:", dir_path)
    return(dir_path, brt_path, coeff_path)


def load_data(dir_path):
    """
    TODO: add description
    :param dir_path:
    :return:
    """
    # load train data
    train_occCovars = pd.read_csv(dir_path + "train_occCovars.csv", header=None)
    train_detCovars = pd.read_csv(dir_path + "train_detCovars.csv", header=None)
    train_occProbs = pd.read_csv(dir_path + "train_occProbs.csv", header=None)
    train_detProbs = pd.read_csv(dir_path + "train_detProbs.csv", header=None)
    train_Y = pd.read_csv(dir_path + "train_detHists.csv", header=None)

    # load validation data
    valid_occCovars = pd.read_csv(dir_path + "valid_occCovars.csv", header=None)
    valid_detCovars = pd.read_csv(dir_path + "valid_detCovars.csv", header=None)
    valid_occProbs = pd.read_csv(dir_path + "valid_occProbs.csv", header=None)
    valid_detProbs = pd.read_csv(dir_path + "valid_detProbs.csv", header=None)
    valid_Y = pd.read_csv(dir_path + "valid_detHists.csv", header=None)

    # load test data
    test_occCovars = pd.read_csv(dir_path + "test_occCovars.csv", header=None)
    test_detCovars = pd.read_csv(dir_path + "test_detCovars.csv", header=None)
    test_occProbs = pd.read_csv(dir_path + "test_occProbs.csv", header=None)
    test_detProbs = pd.read_csv(dir_path + "test_detProbs.csv", header=None)
    test_Y = pd.read_csv(dir_path + "test_detHists.csv", header=None)

    x_dim = train_occCovars.shape[1]
    w_dim = train_detCovars.shape[1]
    train_nSite = train_occCovars.shape[0]
    valid_nSite = valid_occCovars.shape[0]
    test_nSite = test_occCovars.shape[0]
    k = int(train_detCovars.shape[0]/train_nSite)

    return(x_dim, w_dim, k,\
           train_occCovars, train_detCovars, train_occProbs, train_detProbs, train_Y,\
           valid_occCovars, valid_detCovars, valid_occProbs, valid_detProbs, valid_Y,\
           test_occCovars, test_detCovars, test_occProbs, test_detProbs, test_Y)


def load_coeffs(coeff_path):
    """
    TODO: add description
    :param coeff_path:
    :return:
    """
    occCoeffs = pd.read_csv(coeff_path + "occCoeffs.csv", header=None)
    detCoeffs = pd.read_csv(coeff_path + "detCoeffs.csv", header=None)
    return(occCoeffs, detCoeffs)


def sythetic_sanity_check(rho, x_dim, w_dim, occCoeffs, detCoeffs, \
                          train_occCovars, train_detCovars,\
                          train_occProbs, train_detProbs):
    """
    TODO: add description
    :param rho:
    :param x_dim:
    :param w_dim:
    :param occCoeffs:
    :param detCoeffs:
    :param train_occCovars:
    :param train_detCovars:
    :param train_occProbs:
    :param train_detProbs:
    :return:
    """
    occCoeffs = occCoeffs[1:]
    detCoeffs = detCoeffs[1:]

    idx = randint(0, train_occCovars.shape[0]-1)
    if rho == 0:
        tmp1 = np.array(torch.sigmoid(torch.tensor(\
               np.dot(np.array(train_occCovars)[idx,], \
               np.array(occCoeffs).flatten()), dtype=torch.float64)))
        tmp2 = train_occProbs.to_numpy()[idx]
        tmp3 = np.array(torch.sigmoid(torch.tensor(\
               np.dot(np.array(train_detCovars)[idx,], \
               np.array(detCoeffs).flatten()), dtype=torch.float64)))
        tmp4 = train_detProbs.to_numpy()[idx]
    else:
        tmp1 = np.array(torch.sigmoid(torch.tensor(\
               np.dot(np.array(train_occCovars)[idx,]**2, \
               np.array(occCoeffs).flatten()), dtype=torch.float64)))
        tmp2 = train_occProbs.to_numpy()[idx]
        tmp3 = np.array(torch.sigmoid(torch.tensor(\
               np.dot(np.array(train_detCovars)[idx,]**2, \
                      np.array(detCoeffs).flatten()), dtype=torch.float64)))
        tmp4 = train_detProbs.to_numpy()[idx]

    assert np.round(tmp1.item(),5) == np.round(tmp2[0],5)
    assert np.round(tmp3.item(),5) == np.round(tmp4[0],5)


def data_convert(train_occCovars, train_detCovars, train_Y,\
                valid_occCovars, valid_detCovars, valid_Y,\
                test_occCovars, test_detCovars, test_Y):
    """
    TODO: add description
    :param train_occCovars:
    :param train_detCovars:
    :param train_Y:
    :param valid_occCovars:
    :param valid_detCovars:
    :param valid_Y:
    :param test_occCovars:
    :param test_detCovars:
    :param test_Y:
    :return:
    """

    train_nSite = train_occCovars.shape[0]
    valid_nSite = valid_occCovars.shape[0]
    test_nSite = test_occCovars.shape[0]
    k = int(train_detCovars.shape[0]/train_nSite)
    w_dim = train_detCovars.shape[1]

    x_train = np.array(train_occCovars)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    w_train = np.array(train_detCovars)
    w_train = torch.tensor(w_train.reshape(train_nSite, k, w_dim), \
                           dtype=torch.float32)
    y_train = torch.tensor(np.array(train_Y).reshape(train_nSite, k), \
                           dtype=torch.float32)

    x_valid = np.array(valid_occCovars)
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    w_valid = np.array(valid_detCovars)
    w_valid = torch.tensor(w_valid.reshape(valid_nSite, k, w_dim), \
                           dtype=torch.float32)
    y_valid = torch.tensor(np.array(valid_Y).reshape(valid_nSite, k), \
                           dtype=torch.float32)

    x_test = np.array(test_occCovars)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    w_test = np.array(test_detCovars)
    w_test = torch.tensor(w_test.reshape(test_nSite, k, w_dim), \
                          dtype=torch.float32)
    y_test = torch.tensor(np.array(test_Y).reshape(test_nSite, k), \
                          dtype=torch.float32)

    return(x_train, w_train, y_train, \
           x_valid, w_valid, y_valid, \
           x_test, w_test, y_test)


# OR2020 bird species datasets

def load_real_path(species_name):
    """
    TODO: add description
    :param species_name:
    :return:
    """
    dir_path = "../data/OR2020/" + species_name + "/"
    print("data path:", dir_path)
    return(dir_path)


def load_real_data(dir_path, test_fold):
    """
    TODO: add description
    :param dir_path:
    :param test_fold:
    :return:
    """
    if test_fold == 1:
        train_path = dir_path + "f2"
        valid_path = dir_path + "f3"
        test_path = dir_path + "f1"
    elif test_fold == 2:
        train_path = dir_path + "f3"
        valid_path = dir_path + "f1"
        test_path = dir_path + "f2"
    else:
        train_path = dir_path + "f1"
        valid_path = dir_path + "f2"
        test_path = dir_path + "f3"

    # load train data
    train_occCovars = pd.read_csv(train_path + "_occCovars.csv", header=None)
    train_detCovars = pd.read_csv(train_path + "_detCovars.csv", header=None)
    train_Y = pd.read_csv(train_path + "_detHists.csv", header=None)

    # load validation data
    valid_occCovars = pd.read_csv(valid_path + "_occCovars.csv", header=None)
    valid_detCovars = pd.read_csv(valid_path + "_detCovars.csv", header=None)
    valid_Y = pd.read_csv(valid_path + "_detHists.csv", header=None)

    # load test data
    test_occCovars = pd.read_csv(test_path + "_occCovars.csv", header=None)
    test_detCovars = pd.read_csv(test_path + "_detCovars.csv", header=None)
    test_Y = pd.read_csv(test_path + "_detHists.csv", header=None)

    x_dim = train_occCovars.shape[1]
    w_dim = train_detCovars.shape[1]
    train_nSite = train_occCovars.shape[0]
    k = int(train_detCovars.shape[0]/train_nSite)

    return(x_dim, w_dim, k,\
           train_occCovars, train_detCovars, train_Y,\
           valid_occCovars, valid_detCovars, valid_Y,\
           test_occCovars, test_detCovars, test_Y)


def real_data_convert(train_occCovars, train_detCovars, train_Y,\
                valid_occCovars, valid_detCovars, valid_Y,\
                test_occCovars, test_detCovars, test_Y):
    """
    TODO: add description
    :param train_occCovars:
    :param train_detCovars:
    :param train_Y:
    :param valid_occCovars:
    :param valid_detCovars:
    :param valid_Y:
    :param test_occCovars:
    :param test_detCovars:
    :param test_Y:
    :return:
    """

    train_nSite = train_occCovars.shape[0]
    valid_nSite = valid_occCovars.shape[0]
    test_nSite = test_occCovars.shape[0]
    k = int(train_detCovars.shape[0]/train_nSite)
    w_dim = train_detCovars.shape[1]

    x_train = np.array(train_occCovars)
    x_train = preprocessing.scale(x_train)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    w_train = np.array(train_detCovars)
    w_train = preprocessing.scale(w_train)
    w_train = torch.tensor(w_train.reshape(train_nSite, k, w_dim), \
                           dtype=torch.float32)
    y_train = torch.tensor(np.array(train_Y).reshape(train_nSite, k), \
                           dtype=torch.float32)

    x_valid = np.array(valid_occCovars)
    x_valid = preprocessing.scale(x_valid)
    x_valid = torch.tensor(x_valid, dtype=torch.float32)
    w_valid = np.array(valid_detCovars)
    w_valid = preprocessing.scale(w_valid)
    w_valid = torch.tensor(w_valid.reshape(valid_nSite, k, w_dim), \
                           dtype=torch.float32)
    y_valid = torch.tensor(np.array(valid_Y).reshape(valid_nSite, k), \
                           dtype=torch.float32)

    x_test = np.array(test_occCovars)
    x_test = preprocessing.scale(x_test)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    w_test = np.array(test_detCovars)
    w_test = preprocessing.scale(w_test)
    w_test = torch.tensor(w_test.reshape(test_nSite, k, w_dim), \
                          dtype=torch.float32)
    y_test = torch.tensor(np.array(test_Y).reshape(test_nSite, k), \
                          dtype=torch.float32)

    return(x_train, w_train, y_train, \
           x_valid, w_valid, y_valid, \
           x_test, w_test, y_test)


def obtain_bird_data(dir_path, fold):
    """
    TODO: add description
    :param dir_path:
    :param fold:
    :return:
    """
    x_dim, w_dim, k,\
    train_occCovars, train_detCovars, train_Y,\
    valid_occCovars, valid_detCovars, valid_Y,\
    test_occCovars, test_detCovars, test_Y = load_real_data(dir_path, fold)

    x_train, w_train, y_train, x_valid, w_valid, y_valid, \
    x_test, w_test, y_test = real_data_convert(\
                                train_occCovars, train_detCovars, train_Y,\
                                valid_occCovars, valid_detCovars, valid_Y,
                                test_occCovars, test_detCovars, test_Y)

    return(x_dim, w_dim, k, x_train, w_train, y_train, \
           x_valid, w_valid, y_valid, x_test, w_test, y_test)

